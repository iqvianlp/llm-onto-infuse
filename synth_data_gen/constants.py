# This file contains OPEN AI account constants
import os
from dotenv import load_dotenv

# load env variables from .env if any
load_dotenv()

# Azure Cloud settings
TENANT_ID = os.environ.get("TENANT_ID", "ADD_YOUR_TENANT_ID")
INTERACTIVE_CLIENT_ID = os.environ.get("INTERACTIVE_CLIENT_ID", "ADD_YOUR_INTERACTIVE_CLIENT_ID")
AZURE_LOGIN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
SCOPE_INTERACTIVE_CLI = f"api://{INTERACTIVE_CLIENT_ID}/.default"
SCOPE_INTERACTIVE_BROWSER = f"api://{INTERACTIVE_CLIENT_ID}/token"

# Client secrets
NON_INTERACTIVE_CLIENT_ID = os.environ.get("INTERACTIVE_CLIENT_ID", "ADD_YOUR_NON_INTERACTIVE_CLIENT_ID")
SERVICE_PRINCIPAL = os.environ.get("SERVICE_PRINCIPAL", "ADD_YOUR_SERVICE_PRINCIPAL")
SERVICE_PRINCIPAL_SECRET = os.environ.get("SERVICE_PRINCIPAL_SECRET", "ADD_YOUR_SERVICE_PRINCIPAL_SECRET")
SCOPE_NON_INTERACTIVE = f"api://{NON_INTERACTIVE_CLIENT_ID}/.default"

# OpenAI
OPENAI_LOG = os.environ.get("OPENAI_LOG", "info")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ADD_YOUR_OPENAI_API_KEY")
OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE", "ADD_YOUR_OPENAI_API_TYPE")
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "ADD_YOUR_OPENAI_API_VERSION")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "ADD_YOUR_OPENAI_API_BASE")

# Global vars
AZURE_SUBSCRIPTION_ID = "ADD_YOUR_AZURE_SUBSCRIPTION_ID"
OPENAI_ACCOUNT_NAME = "ADD_YOUR_OPENAI_ACCOUNT_NAME"
AZURE_RG_NAME = "ADD_YOUR_AZURE_RG_NAME"
