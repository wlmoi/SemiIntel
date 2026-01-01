# GitHub Actions Workflows

This project includes automated workflows for testing and deployment.

## Available Workflows

### 1. Test Workflow (`test.yml`)

**Triggers:** Push to main/develop, Pull requests

**What it does:**
- Tests on Python 3.8, 3.9, and 3.10
- Lints code with flake8
- Verifies Python syntax
- Checks dependencies
- Runs optional tests

**Status:** ✅ Automatically runs on every push

---

### 2. Azure Web App Deployment (`azure-webapps-python.yml`)

**Triggers:** Push to main branch

**Requirements:**
1. Azure Web App created
2. Azure Service Principal with contributor role
3. GitHub Secrets configured

**Setup Steps:**

1. **Create Azure Web App**
   ```bash
   # In Azure Portal or Azure CLI
   az webapp create --resource-group <resource-group> \
     --plan <app-service-plan> \
     --name semiintel-app \
     --runtime "PYTHON:3.8"
   ```

2. **Create Service Principal**
   ```bash
   az ad sp create-for-rbac --name "semiintel-github-actions" \
     --role contributor \
     --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group> \
     --sdk-auth
   ```
   
   This outputs JSON with credentials you'll need.

3. **Add GitHub Secrets**
   
   Go to: Repository → Settings → Secrets and variables → Actions
   
   Add these secrets:
   - `AZURE_CLIENT_ID` - Application (client) ID from service principal
   - `AZURE_CLIENT_SECRET` - Client secret from service principal  
   - `AZURE_TENANT_ID` - Directory (tenant) ID from service principal
   - `AZURE_SUBSCRIPTION_ID` - Your Azure subscription ID

4. **Update Workflow**
   ```yaml
   env:
     AZURE_WEBAPP_NAME: your-actual-app-name
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Configure Azure deployment"
   git push origin main
   ```

**Workflow Steps:**
- Checks out code
- Sets up Python 3.8
- Installs dependencies
- Uploads artifacts
- Logs in to Azure using service principal
- Deploys to Azure Web App

---

## Streamlit Cloud Deployment

For Streamlit Cloud (recommended for this project):

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Deploy from repository
4. No workflows needed - automatic!

---

## Troubleshooting

### Missing Azure Credentials Error
**Issue:** `Error: Deployment Failed, Error: No credentials found`

**Solution:** Add the Azure Login step with service principal credentials:
1. Create a service principal in Azure
2. Add GitHub secrets: `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_SUBSCRIPTION_ID`, `AZURE_TENANT_ID`
3. Push updated workflow

### Python Cache Error (400)
**Issue:** `Cache service responded with 400`

**Solution:** Already fixed in latest version of setup-python@v5

### Spacy Installation Error
**Issue:** `ERROR: No matching distribution found for numpy<3.0.0,>=2.0.0`

**Solution:** Already fixed - requirements.txt pins spacy<3.7.0 for Python 3.8 compatibility

### Deployment Fails
**Check:**
1. Dependencies install correctly: `pip install -r requirements.txt`
2. app.py has no syntax errors: `python -m py_compile app.py`
3. Secret is configured correctly
4. App name matches your Azure Web App

### Slow Deployments
- First deployment is slower (5-10 min)
- Subsequent deployments use cache (2-3 min)
- Check Actions tab for real-time progress

---

## Local Testing

Test locally before pushing:

```bash
# Run syntax check
python -m py_compile app.py modules/*.py

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## Configuration Files

- `.github/workflows/test.yml` - CI/CD testing
- `.github/workflows/azure-webapps-python.yml` - Azure deployment
- `.streamlit/config.toml` - Streamlit settings
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies

All workflows use:
- Python 3.8+ (compatible with requirements)
- Ubuntu latest runner
- Caching for faster builds
