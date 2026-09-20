/**
 * ONE set of value-coercion and additional-arg rules for the engine command line.
 *
 * src/main/sessions.ts BUILDS the argv the engine is launched with;
 * SessionSettings.tsx renders the "CLI Command Preview" of that same argv. Both
 * carried byte-identical private copies of these helpers AND of the 66-entry
 * value-flag list. That is the shape that has already lied to a user in this
 * project: when the two sides disagree the preview shows one command and the
 * launcher runs another, and nothing catches it because each side is covered by
 * its own tests. (Checked at extraction time: the two lists were still in
 * agreement — this removes the opportunity, it does not fix a live divergence.)
 *
 * `filterAdditionalArgs` is the sharpest of them: it decides which user-typed
 * extra flags survive into the launch. A duplicate flag is not harmless —
 * argparse takes the LAST occurrence, so a stray user flag silently overrides a
 * computed family default.
 */

export const ADDITIONAL_ARG_VALUE_FLAGS = new Set([
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--block-disk-cache-max-percent',
  '--allowed-origins',
  '--api-key',
  '--ssl-certfile',
  '--ssl-keyfile',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--chat-template',
  '--chat-template-kwargs',
  '--completion-batch-size',
  '--cluster-secret',
  '--default-enable-thinking',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-temperature',
  '--default-top-k',
  '--default-top-p',
  '--distributed-mode',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--embedding-model',
  '--flash-moe-io-split',
  '--flash-moe-prefetch',
  '--flash-moe-slot-bank',
  '--host',
  '--inference-endpoints',
  '--image-mode',
  '--image-quantize',
  '--kv-cache-group-size',
  '--kv-cache-quantization',
  '--log-level',
  '--max-cache-blocks',
  '--max-num-seqs',
  '--max-prompt-tokens',
  '--max-tokens',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--mflux-class',
  '--model-family',
  '--num-draft-tokens',
  '--native-mtp-depth',
  '--native-mtp-depth-policy',
  '--native-mtp-sampling-policy',
  '--omni-backend',
  '--paged-cache-block-size',
  '--pld-summary-interval',
  '--port',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--prefix-cache-max-bytes',
  '--prefix-cache-size',
  '--rate-limit',
  '--reasoning-parser',
  '--served-model-name',
  '--smelt-experts',
  '--speculative-model',
  '--ssm-state-cache-mb',
  '--ssm-state-cache-size',
  '--stream-interval',
  '--timeout',
  '--tool-call-parser',
  '--uds',
  '--wake-timeout',
  '--vision-memory-cache-size',
  '--worker-nodes',
])

/** Finite and strictly positive, else undefined. */
export function finitePositiveNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

/** Finite and non-negative, else undefined. Zero is meaningful for several flags. */
export function finiteNonNegativeNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

/** Positive integer, floored, never below 1. */
export function finitePositiveInteger(value: unknown): number | undefined {
  const number = finitePositiveNumber(value)
  return number == null ? undefined : Math.max(1, Math.floor(number))
}

/** Non-negative integer, including 0 (used for --rate-limit 0 = unlimited). */
export function finiteNonNegativeInteger(value: unknown): number | undefined {
  const number = finiteNonNegativeNumber(value)
  return number == null ? undefined : Math.max(0, Math.floor(number))
}

/** Empty or "loopback" means the engine default (localhost pages only). */
export function corsOriginsLaunchArgs(corsOrigins: string | undefined): string[] {
  const value = (corsOrigins ?? '').trim()
  if (!value || value === 'loopback') return []
  return ['--allowed-origins', value]
}

export function generateVmlxApiKey(): string {
  const bytes = new Uint8Array(18)
  crypto.getRandomValues(bytes)
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return `vmlx_${btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')}`
}

/**
 * Split a user-supplied "additional arguments" string into argv tokens, dropping
 * any flag the caller already emits itself. A blocked flag written as
 * `--flag value` also consumes its value token; `--flag=value` carries it inline.
 */
export function filterAdditionalArgs(raw: string | undefined, blockedFlags: Set<string>): string[] {
  if (!raw?.trim()) return []
  const extra = raw.trim().split(/\s+/).filter(Boolean)
  const filtered: string[] = []
  for (let i = 0; i < extra.length; i++) {
    const flag = extra[i]
    const flagName = flag.includes('=') ? flag.slice(0, flag.indexOf('=')) : flag
    if (blockedFlags.has(flagName)) {
      if (flag === flagName && ADDITIONAL_ARG_VALUE_FLAGS.has(flagName)) i++
      continue
    }
    filtered.push(flag)
  }
  return filtered
}
