# PReDUS: A Privacy Requirements Detector from User Stories

PReDUS is a web application that want to provide some insights about privacy requirements.

The tool is highly recommended for stakeholders with little experience in the identification of privacy content.

The algorithm behind PReDUS requires an US as input, whose format is as the following:
```
As a project manager I want to access to data about my colleagues progress so I can better report our success and failures.
```
The output of this application is the prevision of the deep learning algorithm developed to detect privacy content, supported by some information about the words higly related to privacy matters and which kind of privacy they're related to.

## Installation & Configuration

In order to run the web application, all you need is Python.

The application has been tested till 3.9.9 Python version. You can download it [here](https://www.python.org/downloads/).

Once Python is installed, you need to install all the packages listed into ```requirements.txt``` file. You can do it with the following command:

```bash
pip install -r requirements.txt
```

Or install the packages one at time. Example:

```bash
pip install pandas
```

Please, be carefull with version of some packages, or you'll have some issues by using the tool.

Once all the packages are installed, run ```Privacy_Detector_App.py``` script to use the tool. You can do it by using the following command:

```bash
python Privacy_Detector_App.py
```

Once the script has ended you'll get a result like the following one:
```bash
 * Debugger is active!
 * Debugger PIN: XXX-XXX-XXX
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Open your browser and type the url provided. (In the example: http://127.0.0.1:5000/)