# Port of the tool service (sourced by start_services.sh and register_mcp.sh).
#
#   IVF_TOOLS_PORT set            -> used as it is
#   DRESSAGE_BLACKBOX_SLOT_ID set -> $IVF_PORT_BASE + slot (default base 20000): Dressage local_bwrap
#                                    slots share the host network, so each slot needs its own port or
#                                    one case would reach another case's service (held out for a
#                                    different case)
#   otherwise                     -> 9201 (one sandbox per network, e.g. E2B)
# The tool specs (task/tools/*.json) name 9201; register_mcp.sh maps it to this port.
if [ -z "${IVF_TOOLS_PORT:-}" ]; then
  if [ -n "${DRESSAGE_BLACKBOX_SLOT_ID:-}" ]; then
    IVF_TOOLS_PORT=$(( ${IVF_PORT_BASE:-20000} + DRESSAGE_BLACKBOX_SLOT_ID ))
  else
    IVF_TOOLS_PORT=9201
  fi
fi
export IVF_TOOLS_PORT
