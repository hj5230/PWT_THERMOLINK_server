# THERMOLINK server

## Brief description

This is the server of THERMOLINK project, it is built with Python framework Flask. It is meant to provide backend support for THERMOLINK clients (which shall be installed at user's devices). All external modules should be attached in this project and provide services via RESTful APIs to client side.

## How to start

The THERMOLINK server project as a whole is development in Python virtual environment to form an isolated development environment and more efficient dependencies management.

NOTE! This project is developed and should be running on **Ubuntu**, there is no guarantee the project works on other platform. Please find the release, system, and other version information at the end of this file.

Below are instructions on how to use the THERMOLINK server project:

1. install Python3, pip3, and venv (Optional: skip if you already have Python3, pip3, and installed)

```bash
sudo apt install python3 &&
sudo apt install python3-pip &&
sudo pip3 install virtualenv
```
2. activate the virtual environment

```bash
source venv/bin/activate
```

3. start the flask app

```bash
flask run
```

4. when you done with the development

```bash
deactivate
```

## Environment specification
OS: Ubuntu 22.04.3 LTS

Linux *username* 6.5.0-14-generic #14~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC *timestamp* x86_64 x86_64 x86_64 GNU/Linux

