import setuptools

# Define list of individual packages to install here (with specific versions if desired - recommmended)
# NOTE: Obtain list with pipreqs (pip install pipreqs) and run 'pipreqs .'; May need to change '==' to '>=' where necessary
with open('requirements.txt', 'r') as f:
   install_requires = [line.strip() for line in f.readlines() if not line.strip().startswith('#')]

setuptools.setup(
    name="data-science-packages",
    author="Brett Allen",
    author_email="brettallen777@gmailcom",
    version="0.0.2",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    include_package_data=True,
)

