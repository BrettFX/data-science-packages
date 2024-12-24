__author__ = 'Brett Allen (brettallen777@gmail.com)'

import boto3
from botocore.exceptions import ClientError
import re
import pandas as pd
import json
import os
import folium
from folium.plugins import MarkerCluster, Geocoder
from abc import ABC, abstractmethod
from tqdm import tqdm
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_aws import ChatBedrock
import contextlib
from io import StringIO
import sys
import datetime
from typing import Union, List, Callable

# Load default response structure
DEFAULT_RESPONSE_STRUCTURE = 'NOT AVAILABLE'

def process_response(text: str) -> dict:
    """
    Process LLM response and parse JSON from part of response via regular expresions.

    Args:
        text (str): LLM response.

    Returns:
        dict: Parsed JSON data from LLM response.
    """
    if '{' not in text:
        return text

    json_idx = text.index('{')
    m = re.search(r'\{[\s\S]+\}', text[json_idx:])
    response_text = m.group(0).replace('\n', '')
    response_text = re.sub(r'\\+([\"\'])', '', response_text)                  # Remove escaped quotes
    response_text = re.sub(r'\\+([\[\]\(\)\{\}])', r'\g<1>', response_text)    # Remove escaped special characters
    response_text = re.sub('None', 'null', response_text, flags=re.IGNORECASE) # Convert None values to null for json support
    response_text = response_text.replace('\\', '')
    return json.loads(response_text)

def create_json_safe_string(text: str) -> str:
    """
    Escapes special characters to create a JSON-safe string.

    This function takes a string input and escapes any special characters
    that are not valid in JSON. This is useful for converting a string that
    may contain special characters that are not valid in JSON into a string
    that is safe to use in a JSON object.

    Parameters:
        text (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    text = re.sub(r'[\[\]\(\)\{\}\"\']+', r'\\\g<0>', text)
    return text

# Abstract Bedrock Classifier class
class BedrockClassifier(ABC):
    def __init__(self, model_id: str, prompt_template: str, col_map: dict={}, response_structure: dict={}, extraneous_response_keys: List[str]=[], 
                 format_mapper: dict={}, region: str='us-east-1', debug: bool=False, **model_kwargs):
        self.model_id = model_id
        self.model_name = re.sub(r'[\.\-\:]+', '_', model_id)
        self.region = region
        self.model_kwargs = model_kwargs
        
        self.bedrock_runtime = boto3.client(
          service_name='bedrock-runtime',
          region_name=region
        )
        
        # Configurations
        self.col_map = col_map
        self.prompt_template = prompt_template
        self.response_structure = response_structure
        self.extraneous_response_keys = extraneous_response_keys
        self.debug = debug

        # Validate format mapper to ensure keys are strings and values are functions (callable)
        self.format_mapper = {}
        if format_mapper:
            for col, value in format_mapper.items():
                if not isinstance(value, Callable):
                    print(f'[WARNING] Invalid format function for column, "{col}". Will not be able to apply formatting for this column.')
                    continue
                self.format_mapper[col] = value
        
    def process_response(self, text: str) -> dict:
        """
        Process LLM response and parse JSON from part of response via regular expresions.

        Args:
            text (str): LLM response.

        Returns:
            dict: Parsed JSON data from LLM response.
        """
        if '{' not in text:
            return text

        json_idx = text.index('{')
        m = re.search(r'\{[\s\S]+\}', text[json_idx:])
        response_text = m.group(0).replace('\n', '')
        response_text = re.sub(r'\\+([\"\'])', '', response_text)                  # Remove escaped quotes
        response_text = re.sub(r'\\+([\[\]\(\)\{\}])', r'\g<1>', response_text)    # Remove escaped special characters
        response_text = re.sub('None', 'null', response_text, flags=re.IGNORECASE) # Convert None values to null for json support
        response_text = response_text.replace('\\', '')
        return json.loads(response_text)
    
    def create_json_safe_string(self, text: str) -> str:
        """
        Escapes special characters to create a JSON-safe string.

        This function takes a string input and escapes any special characters
        that are not valid in JSON. 

        Parameters:
            text (str): The string to escape.

        Returns:
            str: The escaped string.
        """
        text = re.sub(r'[\[\]\(\)\{\}\"\']+', r'\\\g<0>', text)
        return text


    def build_prompt_values(self, row: Union[pd.Series, dict]) -> dict:
        """
        Build prompt values from a given row. This method takes a row, which could be a pandas Series or a dictionary,
        and applies any specified formatting to the values, then JSON encodes them. The resulting dictionary is returned.

        Args:
            row (Union[pd.Series, dict]): The row to process.

        Returns:
            dict: A dictionary of prompt values.
        """
        prompt_values = {}

        # Set column mapping if it was not configured
        if not self.col_map:
            cols = list(row.index) if isinstance(row, pd.Series) else list(row.keys())
            self.col_map = { col: col for col in cols }

        for key in self.col_map:
            if key in self.format_mapper:
                prompt_values[key] = self.format_mapper[key](row[self.col_map[key]])
            else:
                prompt_values[key] = row[self.col_map[key]]
            
        return prompt_values
    
    def sanitize_prompt_values(self, prompt_values: dict) -> dict:
        """
        Sanitizes prompt values to ensure they are JSON-safe.

        This method takes a dictionary of prompt values and escapes any special characters
        that are not valid in JSON format. It serializes each value to a JSON string and
        ensures it is safe to use in JSON objects.

        Parameters:
            prompt_values (dict): The dictionary containing prompt values to sanitize.

        Returns:
            dict: A dictionary with JSON-safe prompt values.
        """
        # for key in prompt_values:
        #     prompt_values[key] = self.create_json_safe_string(json.dumps(prompt_values[key], default=str))
        return prompt_values

    @abstractmethod
    def predict(self, row: Union[pd.Series, dict], extraneous_response_keys: List[str]=[], **kwargs) -> dict:
        pass
        
    def predict_batch(self, data: Union[pd.DataFrame, List[dict]], extraneous_response_keys: List[str]=[], **kwargs) -> dict:
        preds = []
        errors = []

        # Set column mapping if it was not configured
        if not self.col_map:
            cols = data.columns if isinstance(data, pd.DataFrame) else list(data[0].keys())
            self.col_map = { col: col for col in cols }

        # Set extraneous response keys, prioritizing overrides as necessary
        extraneous_response_keys = extraneous_response_keys or self.extraneous_response_keys

        with tqdm(total=len(data)) as pbar:
            iterator = data.iterrows() if isinstance(data, pd.DataFrame) else enumerate(data)
            for idx, row in iterator:
                name = idx if not isinstance(row, pd.Series) else row.name

                try:
                    pred = self.predict(row, extraneous_response_keys, **kwargs)
                    preds.append({ 'id': idx, 'name': name, **pred })
                except Exception as e:
                    if self.debug:
                        print(str(e))
                        
                    # NOTE: Intentionally not applying formatting here for debugging purposes
                    errors.append({
                        'id': idx,
                        'name': name,
                        'error': str(e),
                        **{key: row[self.col_map[key]] for key in self.col_map},
                    })
                
                pbar.update(1)

        return dict(
            preds=pd.DataFrame.from_dict(preds), 
            errors=errors,
        )

class BedrockLangchainClassifier(BedrockClassifier):
    def __init__(self, model_id: str, prompt_template: str, langchain_agent_data_path: Union[List[str], str], response_structure: Union[str, dict]=None, 
                 extraneous_response_keys: List[str]=[], col_map: dict = {}, format_mapper: dict={}, region: str = 'us-east-1', 
                 model_name_suffix: str='_langchain', debug: bool = False, **model_kwargs):
        super().__init__(
            model_id=model_id,
            prompt_template=prompt_template,
            col_map=col_map,
            response_structure=response_structure,
            extraneous_response_keys=extraneous_response_keys,
            format_mapper=format_mapper,
            region=region,
            debug=debug,
            **model_kwargs
        )

        if model_name_suffix:
            self.model_name += model_name_suffix
        self.langchain_agent_data_path = langchain_agent_data_path

        # Load response structure or use default
        self.response_structure = response_structure or DEFAULT_RESPONSE_STRUCTURE            

        # Initialize bedrock LLM
        self.llm = ChatBedrock(
            model_id=model_id,
            client=self.bedrock_runtime,
            model_kwargs={ 'temperature': 0.0, **self.model_kwargs, **model_kwargs}
        )

        # Initialize csv agent
        self.agent = create_csv_agent(
            self.llm,
            path=self.langchain_agent_data_path,
            verbose=debug, # NOTE: Set verbose True to see thoughts and actions process
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
    
    def predict(self, row: Union[pd.Series, dict], extraneous_response_keys: List[str]=[], **kwargs) -> dict:
        # Set extraneous response keys, prioritizing overrides as necessary
        extraneous_response_keys = extraneous_response_keys or self.extraneous_response_keys

        # Set column mapping if it was not configured
        if not self.col_map:
            cols = list(row.index) if isinstance(row, pd.Series) else list(row.keys())
            self.col_map = { col: col for col in cols }
        
        # Use column map to get features
        # Apply formatting as necessary
        prompt_values = self.build_prompt_values(row)
        
        prompt_criteria = dict(
            response_structure=self.response_structure,
            **self.sanitize_prompt_values({**prompt_values})
        )

        # Use features to build prompt and messages for LLM
        prompt = self.prompt_template.format(**prompt_criteria)

        # Invoke langchain agent with prompt and redirect stdout accordingly
        with contextlib.redirect_stdout(sys.stdout if self.debug else StringIO()):
            results = self.agent.invoke(prompt, **{**self.model_kwargs, **kwargs})

        # Process the results and build prediction
        pred = self.process_response(results['output'])

        # Exclude csv keys and response_structure from pred
        # NOTE: Providing default None value prevents errors when respective key does not exist
        prompt_criteria.pop('csv_path', None)
        prompt_criteria.pop('csv_metadata', None)
        prompt_criteria.pop('response_structure', None)
        for key in extraneous_response_keys:
            prompt_criteria.pop(key, None)

        pred = { **prompt_values, **pred }
        
        return pred

class MistralBedrockClassifier(BedrockClassifier):
    def __init__(self, prompt_template: str, model_id: str="mistral.mixtral-8x7b-instruct-v0:1", col_map: dict={}, response_structure: dict={}, 
                 extraneous_response_keys: List[str]=[], format_mapper: dict={}, region: str='us-east-1', debug: bool=False, **model_kwargs):
        super().__init__(
            model_id=model_id,
            prompt_template=prompt_template,
            col_map=col_map,
            response_structure=response_structure,
            extraneous_response_keys=extraneous_response_keys,
            format_mapper=format_mapper,
            region=region,
            debug=debug,
            **model_kwargs
        )
    
    def predict(self, row: Union[pd.Series, dict], extraneous_response_keys: List[str]=[], **kwargs) -> dict:
        accept = 'application/json'
        content_type = 'application/json'

        # Set extraneous response keys, prioritizing overrides as necessary
        extraneous_response_keys = extraneous_response_keys or self.extraneous_response_keys

        # Set column mapping if it was not configured
        if not self.col_map:
            cols = list(row.index) if isinstance(row, pd.Series) else list(row.keys())
            self.col_map = { col: col for col in cols }
        
        # Use column map to get features
        prompt_values = self.build_prompt_values(row)
        
        prompt_criteria = dict(
            response_structure=self.response_structure,
            **self.sanitize_prompt_values({**prompt_values})
        )

        # Use features to build prompt and messages for LLM
        prompt = self.prompt_template.format(**prompt_criteria)
        
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.1,
            **self.model_kwargs,
            **kwargs,            
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            accept=accept,
            contentType=content_type,
            body=body,
        )
        
        response_body = json.loads(response.get('body').read())
        response = response_body['outputs'][0]['text']
        # print(response)
        
        # Process response
        pred = self.process_response(response)

        # Remove extraneous response keys as desired
        for key in extraneous_response_keys:
            prompt_criteria.pop(key, None)

        pred = { **prompt_values, **pred }
        
        return pred

class LlamaBedrockClassifier(BedrockClassifier):
    def __init__(self, prompt_template: str, model_id: str="meta.llama3-70b-instruct-v1:0", col_map: dict={}, response_structure: dict={}, 
                 extraneous_response_keys: List[str]=[], format_mapper: dict={}, region: str='us-east-1', debug: bool=False, **model_kwargs):
        super().__init__(
            model_id=model_id,
            prompt_template=prompt_template,
            col_map=col_map,
            response_structure=response_structure,
            extraneous_response_keys=extraneous_response_keys,
            format_mapper=format_mapper,
            region=region,
            debug=debug,
            **model_kwargs
        )
    
    def predict(self, row: Union[pd.Series, dict], extraneous_response_keys: List[str]=[], **kwargs) -> dict:
        # Set extraneous response keys, prioritizing overrides as necessary
        extraneous_response_keys = extraneous_response_keys or self.extraneous_response_keys

        # Set column mapping if it was not configured
        if not self.col_map:
            cols = list(row.index) if isinstance(row, pd.Series) else list(row.keys())
            self.col_map = { col: col for col in cols }

        # Use column map to get features
        prompt_values = self.build_prompt_values(row)
        
        prompt_criteria = dict(
            response_structure=self.response_structure,
            **self.sanitize_prompt_values({**prompt_values})
        )

        # Use features to build prompt and messages for LLM
        prompt = self.prompt_template.format(**prompt_criteria)
        
        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.1,
            **self.model_kwargs,
            **kwargs,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)
        response_body = None
        pred = {}

        try:
            # Invoke the model with the request.
            response = self.bedrock_runtime.invoke_model(modelId=self.model_id, body=request)

            # Decode the response body.
            response_body = response["body"].read()
            model_response = json.loads(response_body.decode('utf-8').replace('`', ''))

            # Extract and print the response text.
            response_text = model_response["generation"]
            # print(response_text)
            
            # Process response
            pred = self.process_response(response_text)

            # Remove extraneous response keys as desired
            for key in extraneous_response_keys:
                prompt_criteria.pop(key, None)
            
            pred = { **prompt_values, **pred }

        except (ClientError, Exception) as e:
            if self.debug:
                print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
                print('Attempted request body:')
                print(request)

                if response_body is not None:
                    print('\nResponse body:')
                    print(response_body)
                    
            pred = None
        
        return pred

class ClaudeBedrockClassifier(BedrockClassifier):
    def __init__(self, prompt_template: str, model_id: str="anthropic.claude-3-5-sonnet-20240620-v1:0", col_map: dict={}, response_structure: dict={}, 
                 extraneous_response_keys: List[str]=[], format_mapper: dict={}, region: str='us-east-1', debug: bool=False, **model_kwargs):
        super().__init__(
            model_id=model_id,
            prompt_template=prompt_template,
            col_map=col_map,
            response_structure=response_structure,
            extraneous_response_keys=extraneous_response_keys,
            format_mapper=format_mapper,
            region=region,
            debug=debug,
            **model_kwargs
        )
    
    def predict(self, row: Union[pd.Series, dict], extraneous_response_keys: List[str]=[], **kwargs) -> dict:
        # Set extraneous response keys, prioritizing overrides as necessary
        extraneous_response_keys = extraneous_response_keys or self.extraneous_response_keys

        # Set column mapping if it was not configured
        if not self.col_map:
            cols = list(row.index) if isinstance(row, pd.Series) else list(row.keys())
            self.col_map = { col: col for col in cols }
        
        # Use column map to get features
        prompt_values = self.build_prompt_values(row)
        
        prompt_criteria = dict(
            response_structure=self.response_structure,
            **self.sanitize_prompt_values({**prompt_values})
        )

        # Use features to build prompt and messages for LLM
        prompt = self.prompt_template.format(**prompt_criteria)
        
        # Format the request payload using the model's native structure.
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            **self.model_kwargs,
            **kwargs,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)
        response_body = None
        pred = {}

        try:
            # Invoke the model with the request.
            response = self.bedrock_runtime.invoke_model(modelId=self.model_id, body=request)

            # Decode the response body.
            response_body = response["body"].read()
            model_response = json.loads(response_body.decode('utf-8').replace('`', ''))

            # Extract and print the response text.
            response_text = model_response["content"][0]["text"]

            if self.debug:
                print('\nResponse text:')
                print(response_text)
            
            # Process response
            pred = self.process_response(response_text)

            # Remove extraneous response keys as desired
            for key in extraneous_response_keys:
                prompt_criteria.pop(key, None)

            if self.debug:
                print('\nPrompt values:')
                print(prompt_values)
                print('\nPrediction:')
                print(pred)

            pred = { **prompt_values, **pred }

        except (ClientError, Exception) as e:
            if self.debug:
                print(f"ERROR: {e}")
                print('Attempted request body:')
                print(request)

                if response_body is not None:
                    print('\nResponse body:')
                    print(response_body)
                    
            pred = None
        
        return pred

def run_zero_shot_benchmark(classifier: BedrockClassifier, data: pd.DataFrame, extraneous_response_keys: List[str]=[], output_dir: str='outputs', output_prefix: str='', output_suffix: str='', max_retries: int=5, save: bool=True) -> dict:
    """
    Runs a zero-shot benchmark on a dataset using a specified classifier.

    This function processes the given data using the classifier, resolves any
    errors encountered during predictions, and saves the results and metrics to
    specified output directories. It also provides an analysis of the results,
    including uncertainty distribution and null value counts.

    Args:
        classifier (BedrockClassifier): The classifier to use for predictions.
        data (pd.DataFrame): The input data for which predictions are to be made.
        extraneous_response_keys (List[str], optional): Keys to exclude from the
            response. Defaults to an empty list.
        output_dir (str, optional): Directory to save output files. Defaults to 'outputs'.
        output_prefix (str, optional): Prefix for output filenames. Defaults to ''.
        output_suffix (str, optional): Suffix for output filenames. Defaults to ''.
        max_retries (int, optional): Maximum number of retries for resolving errors.
            Defaults to 5.
        save (bool, optional): Whether to save the results to disk. Defaults to True.

    Returns:
        dict: A dictionary containing the benchmark results, including predictions,
        error counts, metrics, and the paths to the saved files.
    """

    benchmark_results = {
        'original': {},
        'corrections': {},
        'final': {},
        'metrics': {},
    }

    batch_results = classifier.predict_batch(data, extraneous_response_keys)
    preds, errors = batch_results['preds'], batch_results['errors']
    print(f'preds.shape -> {preds.shape}', f'len(errors) -> {len(errors)}')
    benchmark_results['original']['preds.shape'] = preds.shape
    benchmark_results['original']['len(errors)'] = len(errors)

    # Save errors
    errors_path = os.path.join(output_dir, f'{output_prefix}{classifier.model_name}-errors{output_suffix}.json')
    if save:
        os.makedirs(output_dir, exist_ok=True)
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2, default=str)
        print(f'Saved errors to "{errors_path}"')

    # Process errors
    if len(errors) > 0:
        print('Processing errors...')
        batch_results = classifier.predict_batch(errors, extraneous_response_keys)
        resolved_preds, errors = batch_results['preds'], batch_results['errors']

        retries = 0
        while len(errors) > 0 and retries < max_retries:
            print(f'Iteration {retries+1} of {max_retries}: {len(errors)} errors remain.')
            batch_results = classifier.predict_batch(errors, extraneous_response_keys)
            temp_preds, errors = batch_results['preds'], batch_results['errors']
            resolved_preds = pd.concat([resolved_preds, temp_preds], ignore_index=True)
            retries += 1

        print(f'{len(errors)} errors remain after processing errors.')
        benchmark_results['corrections']['preds.shape'] = resolved_preds.shape
        benchmark_results['corrections']['len(errors)'] = len(errors)
        if save:
            with open(errors_path, 'w') as f:
                json.dump(errors, f, indent=2, default=str)
            print(f'Saved updated errors to "{errors_path}"')

        # Combine batch results and resolved results
        preds = pd.concat([preds, resolved_preds], ignore_index=True)
        benchmark_results['final']['preds.shape'] = preds.shape
        benchmark_results['final']['len(errors)'] = len(errors)

    # Save results
    output_path = os.path.join(output_dir, f'{output_prefix}{classifier.model_name}{output_suffix}.csv')
    if save:
        os.makedirs(output_dir, exist_ok=True)
        preds.to_csv(output_path, index=False)
        print(f'Saved classification results to "{output_path}"')

    # Analyze results
    print('Analyzing results...')
    uncertainty_counts = preds['uncertainty'].value_counts()
    analysis_df = pd.DataFrame({
        'count': uncertainty_counts,
        'total': len(preds),
        'ratio': uncertainty_counts / len(preds),
    })
    # print('Uncertainty Distribution:')
    # print(analysis_df)
    benchmark_results['metrics']['uncertainty_distribution'] = analysis_df.to_dict()
    
    # print('\nNull Value Counts:')
    na_counts = preds.isna().sum()
    # print(na_counts)
    benchmark_results['metrics']['na_counts'] = na_counts.to_dict()

    print('\nProcessed Dataset Information:')
    print(preds.info())
    benchmark_results['metrics']['preds.info'] = preds.describe().to_dict()

    print('Zero-Shot Benchmark Test complete!')

    benchmark_results['benchmark_timestamp'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S.%f')
    benchmark_results['is_results_saved'] = save
    benchmark_results['output_paths'] = {
        'output_path': output_path,
        'errors_path': errors_path,
    }

    # Save benchmark results metrics
    benchmark_results_path = os.path.join(output_dir, f'{output_prefix}benchmark_results_{classifier.model_name}{output_suffix}.json')
    if save:
        with open(benchmark_results_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)

    return benchmark_results
