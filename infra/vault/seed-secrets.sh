#!/bin/sh
# One-shot Vault initialiser: enable KV, seed secrets, create AppRole.
# Runs as vault-init container on first docker compose up.
# Idempotent — safe to re-run.
set -e

VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
export VAULT_ADDR
export VAULT_TOKEN="${VAULT_TOKEN:-dev-root-token}"

echo "==> Enabling KV v1 secrets engine at secret/"
vault secrets enable -path=secret kv 2>/dev/null || echo "   (already enabled)"

echo "==> Writing secrets to secret/drift-triage"
vault kv put secret/drift-triage \
  google_api_key="${GOOGLE_API_KEY:-changeme}" \
  postgres_password="${POSTGRES_PASSWORD:-changeme}" \
  promotion_api_key="${PROMOTION_API_KEY:-changeme_changeme_ch}"

echo "==> Enabling AppRole auth"
vault auth enable approle 2>/dev/null || echo "   (already enabled)"

echo "==> Writing drift-triage policy"
vault policy write drift-triage-policy - <<'EOF'
path "secret/drift-triage" {
  capabilities = ["read"]
}
EOF

echo "==> Creating drift-triage AppRole"
vault write auth/approle/role/drift-triage \
  token_policies="drift-triage-policy" \
  secret_id_ttl=87600h \
  token_ttl=1h \
  token_max_ttl=4h

echo "==> AppRole credentials:"
ROLE_ID=$(vault read -field=role_id auth/approle/role/drift-triage/role-id)
SECRET_ID=$(vault write -f -field=secret_id auth/approle/role/drift-triage/secret-id)

echo "   VAULT_ROLE_ID=${ROLE_ID}"
echo "   VAULT_SECRET_ID=${SECRET_ID}"
echo ""
echo "==> Add these to your .env file, then restart service/agent/worker."
echo "Done."
