#!/usr/bin/env bash
# Provision the single Azure VM that hosts the entire ollive stack (CPU-only).
# Also creates an Azure Container Registry the GitHub Actions workflow pushes to.
#
# Usage:
#   ./infra/deploy/deploy.sh \
#       --anthropic-key   <sk-ant-...> \
#       --opencode-key    <oc_...>           # OSS provider (OpenCode-Go) \
#       [--hf-token       <hf-...>] \
#       [--ci-ssh-pubkey  <path-to-pubkey>]  # auto-generated if omitted \
#       [--acr-name       <ollivacrXXXX>]    # auto-generated if omitted \
#       [--repo-url       <https://github.com/you/repo>] \
#       [--branch         <main>] \
#       [--location       <eastus>] \
#       [--vm-size        <Standard_E8s_v5>] \
#       [--rg-name        <ollive-rg>] \
#       [--vm-name        <ollive-vm>] \
#       [--ssh-source     <my-ip-cidr>]

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
REPO_URL_DEFAULT="https://github.com/pranavvp16/LLM-sdk-evals.git"
BRANCH_DEFAULT="main"
LOCATION_DEFAULT="eastus"
VM_SIZE_DEFAULT="Standard_L8aos_v4"         # 8 vCPU / 64 GB / ~$0.75/hr (AMD EPYC, storage-optimized, only family with eastus capacity)
RG_NAME_DEFAULT="ollive-rg"
VM_NAME_DEFAULT="ollive-vm"
SSH_SOURCE_DEFAULT="*"

ANTHROPIC_KEY=""
OPENCODE_KEY=""
HF_TOKEN=""
CI_SSH_PUBKEY_PATH=""
ACR_NAME=""
REPO_URL="$REPO_URL_DEFAULT"
BRANCH="$BRANCH_DEFAULT"
LOCATION="$LOCATION_DEFAULT"
VM_SIZE="$VM_SIZE_DEFAULT"
RG_NAME="$RG_NAME_DEFAULT"
VM_NAME="$VM_NAME_DEFAULT"
SSH_SOURCE="$SSH_SOURCE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --anthropic-key)   ANTHROPIC_KEY="$2"; shift 2 ;;
    --opencode-key)    OPENCODE_KEY="$2"; shift 2 ;;
    --hf-token)        HF_TOKEN="$2"; shift 2 ;;
    --ci-ssh-pubkey)   CI_SSH_PUBKEY_PATH="$2"; shift 2 ;;
    --acr-name)        ACR_NAME="$2"; shift 2 ;;
    --repo-url)        REPO_URL="$2"; shift 2 ;;
    --branch)          BRANCH="$2"; shift 2 ;;
    --location)        LOCATION="$2"; shift 2 ;;
    --vm-size)         VM_SIZE="$2"; shift 2 ;;
    --rg-name)         RG_NAME="$2"; shift 2 ;;
    --vm-name)         VM_NAME="$2"; shift 2 ;;
    --ssh-source)      SSH_SOURCE="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$ANTHROPIC_KEY" ]] && { echo "error: --anthropic-key is required" >&2; exit 1; }
[[ -z "$OPENCODE_KEY"  ]] && { echo "warning: --opencode-key not given; OSS chat/eval will not work until you set OPENCODE_API_KEY in .env on the VM" >&2; }

command -v az >/dev/null 2>&1 || { echo "error: az CLI not found on PATH" >&2; exit 1; }
az account show >/dev/null 2>&1 || { echo "error: not logged in. run 'az login' first." >&2; exit 1; }

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── CI deploy SSH key (auto-generate if not provided) ───────────────────
if [[ -z "$CI_SSH_PUBKEY_PATH" ]]; then
  CI_KEY_PATH="$DEPLOY_DIR/.ci_deploy_key"
  if [[ ! -f "$CI_KEY_PATH" ]]; then
    echo "▸ generating CI deploy SSH keypair at $CI_KEY_PATH"
    ssh-keygen -t ed25519 -N "" -C "ollive-ci-deploy" -f "$CI_KEY_PATH" >/dev/null
  fi
  CI_SSH_PUBKEY_PATH="$CI_KEY_PATH.pub"
fi
[[ -f "$CI_SSH_PUBKEY_PATH" ]] || { echo "error: CI ssh pubkey not found at $CI_SSH_PUBKEY_PATH" >&2; exit 1; }
CI_SSH_PUBKEY=$(<"$CI_SSH_PUBKEY_PATH")

# ── Resource group ───────────────────────────────────────────────────────
echo "▸ ensuring resource group ${RG_NAME} in ${LOCATION}"
az group create --name "$RG_NAME" --location "$LOCATION" --output none

# ── Azure Container Registry ────────────────────────────────────────────
if [[ -z "$ACR_NAME" ]]; then
  ACR_NAME="ollivacr$(date +%s | tail -c 7)"
fi
echo "▸ ensuring container registry ${ACR_NAME} (Basic, admin enabled)"
if ! az acr show --name "$ACR_NAME" --resource-group "$RG_NAME" --output none 2>/dev/null; then
  az acr create --resource-group "$RG_NAME" --name "$ACR_NAME" \
                --sku Basic --location "$LOCATION" --admin-enabled true --output none
fi
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RG_NAME" --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --resource-group "$RG_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --resource-group "$RG_NAME" --query 'passwords[0].value' -o tsv)
INITIAL_IMAGE_TAG="latest"

# ── DNS label / public hostname ─────────────────────────────────────────
DNS_LABEL="ollive-$(date +%s | tail -c 6)"
FQDN="${DNS_LABEL}.${LOCATION}.cloudapp.azure.com"

# ── Templated cloud-init ────────────────────────────────────────────────
TMP_CLOUD_INIT=$(mktemp)
TMP_SUBS=$(mktemp)
trap 'rm -f "$TMP_CLOUD_INIT" "$TMP_SUBS"' EXIT
# Write substitutions to an env file the perl one-liner reads — no quoting hell.
{
  printf '%s\0%s\0' "__REPO_URL__"         "$REPO_URL"
  printf '%s\0%s\0' "__BRANCH__"           "$BRANCH"
  printf '%s\0%s\0' "__PUBLIC_HOSTNAME__"  "$FQDN"
  printf '%s\0%s\0' "__ANTHROPIC_API_KEY__" "$ANTHROPIC_KEY"
  printf '%s\0%s\0' "__HF_TOKEN__"         "$HF_TOKEN"
  printf '%s\0%s\0' "__OPENCODE_API_KEY__" "$OPENCODE_KEY"
  printf '%s\0%s\0' "__ACR_LOGIN_SERVER__" "$ACR_LOGIN_SERVER"
  printf '%s\0%s\0' "__ACR_USERNAME__"     "$ACR_USERNAME"
  printf '%s\0%s\0' "__ACR_PASSWORD__"     "$ACR_PASSWORD"
  printf '%s\0%s\0' "__IMAGE_TAG__"        "$INITIAL_IMAGE_TAG"
  printf '%s\0%s\0' "__CI_SSH_PUBKEY__"    "$CI_SSH_PUBKEY"
} > "$TMP_SUBS"
SUBS_FILE="$TMP_SUBS" TEMPLATE="$DEPLOY_DIR/cloud-init.yaml" OUT="$TMP_CLOUD_INIT" perl -e '
  local $/ = undef;
  open(my $sf, "<", $ENV{SUBS_FILE}) or die "subs: $!";
  my $sb = <$sf>; close $sf;
  my @parts = split /\0/, $sb, -1; pop @parts if @parts && $parts[-1] eq "";
  my %m;
  while (@parts) { my $k = shift @parts; my $v = shift @parts // ""; $m{$k} = $v; }
  open(my $tf, "<", $ENV{TEMPLATE}) or die "template: $!";
  my $t = <$tf>; close $tf;
  for my $k (keys %m) { my $q = quotemeta($k); $t =~ s/$q/$m{$k}/g; }
  open(my $of, ">", $ENV{OUT}) or die "out: $!";
  print $of $t; close $of;
'

# ── VM ───────────────────────────────────────────────────────────────────
if az vm show --resource-group "$RG_NAME" --name "$VM_NAME" --output none 2>/dev/null; then
  echo "▸ VM ${VM_NAME} already exists in ${RG_NAME}; skipping create."
  echo "  Recreate with: az vm delete --resource-group ${RG_NAME} --name ${VM_NAME} --yes"
else
  echo "▸ creating VM ${VM_NAME} (${VM_SIZE}) — this can take ~3 minutes"
  az vm create \
    --resource-group "$RG_NAME" \
    --name "$VM_NAME" \
    --location "$LOCATION" \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
    --size "$VM_SIZE" \
    --admin-username azureuser \
    --generate-ssh-keys \
    --public-ip-sku Standard \
    --public-ip-address-allocation Static \
    --public-ip-address-dns-name "$DNS_LABEL" \
    --os-disk-size-gb 128 \
    --storage-sku Premium_LRS \
    --custom-data "$TMP_CLOUD_INIT" \
    --output none

  echo "▸ opening NSG ports 80, 443 (SSH stays default 22)"
  az vm open-port --resource-group "$RG_NAME" --name "$VM_NAME" --port 80  --priority 1001 --output none
  az vm open-port --resource-group "$RG_NAME" --name "$VM_NAME" --port 443 --priority 1002 --output none

  if [[ "$SSH_SOURCE" != "*" ]]; then
    echo "▸ restricting SSH to ${SSH_SOURCE}"
    NIC_ID=$(az vm show -g "$RG_NAME" -n "$VM_NAME" --query 'networkProfile.networkInterfaces[0].id' -o tsv)
    NSG_NAME=$(az network nic show --ids "$NIC_ID" --query 'networkSecurityGroup.id' -o tsv | awk -F/ '{print $NF}')
    az network nsg rule update --resource-group "$RG_NAME" --nsg-name "$NSG_NAME" --name default-allow-ssh --source-address-prefixes "$SSH_SOURCE" --output none
  fi
fi

PUBLIC_IP=$(az vm show -d --resource-group "$RG_NAME" --name "$VM_NAME" --query publicIps -o tsv)

cat <<EOF

────────────────────────────────────────────────────────────────────────
  ollive deployment kicked off.

  Public URL:        https://${FQDN}
  Public IP:         ${PUBLIC_IP}
  SSH:               ssh azureuser@${PUBLIC_IP}
  ACR registry:      ${ACR_LOGIN_SERVER}
  ACR username:      ${ACR_USERNAME}
  CI ssh key (priv): ${CI_SSH_PUBKEY_PATH%.pub}
  CI ssh key (pub):  ${CI_SSH_PUBKEY_PATH}

  Bootstrap takes ~5–10 min. Watch with:
      ssh azureuser@${PUBLIC_IP} 'sudo tail -f /var/log/cloud-init-output.log'

  When ready:
      ssh azureuser@${PUBLIC_IP} 'sudo cat /opt/ollive/credentials.txt'

  Cost: VM ~\$0.50/hr (${VM_SIZE}) + ACR Basic ~\$0.17/day.
  Tear down: az group delete --name ${RG_NAME} --yes --no-wait

  Set GitHub secrets (one-time):
      gh secret set ACR_LOGIN_SERVER -b '${ACR_LOGIN_SERVER}'
      gh secret set ACR_USERNAME     -b '${ACR_USERNAME}'
      gh secret set ACR_PASSWORD     -b '${ACR_PASSWORD}'
      gh secret set DEPLOY_HOST      -b '${FQDN}'
      gh secret set PUBLIC_HOSTNAME  -b '${FQDN}'
      gh secret set DEPLOY_SSH_KEY < ${CI_SSH_PUBKEY_PATH%.pub}
────────────────────────────────────────────────────────────────────────
EOF
