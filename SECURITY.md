# Security Policy

## API Key Management

This project requires API keys for certain features (e.g., OpenRouter API for AI functionality). To keep your keys secure:

### ✅ DO:
- Store API keys in environment variables
- Use a `.env` file (already in `.gitignore`)
- Load environment variables using `python-dotenv`
- Share example configuration files with placeholder values

### ❌ DON'T:
- Hardcode API keys in source code
- Commit `.env` files to the repository
- Share API keys in issue comments or pull requests
- Include API keys in screenshots or documentation

## Setting Up Your API Keys

### Method 1: Environment Variable (Recommended for Production)

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### Method 2: .env File (Recommended for Development)

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your-api-key-here
```

The `.env` file is already in `.gitignore` and will not be committed.

### Method 3: Streamlit Secrets (For Streamlit Cloud Deployment)

Create `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "your-api-key-here"
```

## Secret Scanning

This repository uses [Gitleaks](https://github.com/gitleaks/gitleaks) to prevent secrets from being committed.

### Running Gitleaks Locally

```bash
# Install gitleaks (macOS)
brew install gitleaks

# Install gitleaks (Linux)
# Download the latest release from: https://github.com/gitleaks/gitleaks/releases/latest
wget https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz
tar -xzf gitleaks_linux_x64.tar.gz

# Scan for secrets
gitleaks detect --source . --verbose

# Scan before committing
gitleaks protect --staged --verbose
```

### Pre-commit Hook (Recommended)

To automatically scan for secrets before each commit, add a pre-commit hook:

```bash
# Create the hook file
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
gitleaks protect --staged --verbose
EOF

# Make it executable
chmod +x .git/hooks/pre-commit
```

## Reporting Security Issues

If you discover a security vulnerability in this repository:

1. **Do NOT** create a public GitHub issue
2. Contact the repository owner directly
3. Provide details about the vulnerability
4. Wait for confirmation before disclosing publicly

## Security Audit Results

Last audit: 2025-12-06

✅ **Status: CLEAN**
- No API keys or secrets found in current files
- No secrets detected in git history
- `.env` properly configured in `.gitignore`
- Gitleaks configuration added for ongoing protection

## Dependencies

Keep dependencies up to date to avoid security vulnerabilities:

```bash
pip install --upgrade -r requirements.txt
```

## Best Practices

1. **Regular Audits**: Run `gitleaks detect` periodically
2. **Review PRs**: Check for accidentally committed secrets
3. **Rotate Keys**: If a key is exposed, rotate it immediately
4. **Limit Scope**: Use API keys with minimum necessary permissions
5. **Monitor Usage**: Check API usage for unexpected activity

## Additional Resources

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [Gitleaks Documentation](https://github.com/gitleaks/gitleaks)
