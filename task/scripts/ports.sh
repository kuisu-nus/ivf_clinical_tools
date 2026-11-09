# Port of the tool service (sourced by start_services.sh and register_mcp.sh).
#
#   IVF_TOOLS_PORT set                         -> used as it is
#   DRESSAGE_BLACKBOX_SLOT_ID a number         -> $IVF_PORT_BASE + slot (default base 20000): Dressage
#                                                 local_bwrap slots share the host network, so each slot
#                                                 needs its own port or one case would reach another
#                                                 case's service (held out for a different case)
#   otherwise (unset, or "e2b" in E2B sandboxes) -> 9201 (one sandbox per network)
# The tool specs (task/tools/*.json) name 9201; register_mcp.sh maps it to this port.
if [ -z "${IVF_TOOLS_PORT:-}" ]; then
  case "${DRESSAGE_BLACKBOX_SLOT_ID:-}" in
    '' | *[!0-9]*) IVF_TOOLS_PORT=9201 ;;
    *) IVF_TOOLS_PORT=$(( ${IVF_PORT_BASE:-20000} + 10#$DRESSAGE_BLACKBOX_SLOT_ID )) ;;
  esac
fi
export IVF_TOOLS_PORT
