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
2. Publish Profile downloaded from Azure Portal
3. GitHub Secret `AZURE_WEBAPP_PUBLISH_PROFILE` configured

**Setup Steps:**

1. Create Azure Web App
   ```
   Go to Azure Portal → Create Web App
   Runtime: Python 3.8
   ```

2. Download Publish Profile
   ```
   Azure Portal → Your Web App → Get Publish Profile
   ```

3. Add GitHub Secret
   ```
   Settings → Secrets and variables → Actions
   Name: AZURE_WEBAPP_PUBLISH_PROFILE
   Value: (paste entire .publishSettings file)
   ```

4. Update Workflow
   ```yaml
   env:
     AZURE_WEBAPP_NAME: your-app-name
   ```

5. Commit and Push
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

### Python Cache Error (400)
**Issue:** `Cache service responded with 400`

**Solution:** Already fixed in latest version of setup-python@v5

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
