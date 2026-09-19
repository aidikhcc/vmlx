import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { InlineToolCall, ToolCallBody } from '../src/renderer/src/components/chat/InlineToolCall'
import { getToolSummary, parseToolArgs } from '../src/renderer/src/components/chat/chat-utils'

function body(toolName: string, args: Record<string, unknown>, isError = false) {
  return renderToStaticMarkup(React.createElement(ToolCallBody, {
    toolName, args, callingDetail: JSON.stringify(args), isError,
    resultDetail: isError ? 'Rejected: expected a string' : 'Result preserved',
  }))
}

describe('untrusted tool arguments remain inspectable', () => {
  const malformed: Array<[string, Record<string, unknown>]> = [
    ['write_file', { path: 'answer.json', content: { count: 1000 } }],
    ['write_file', { path: { unexpected: true }, content: 'text' }],
    ['edit_file', { path: 'a.txt', search_text: {}, replacement_text: 'next' }],
    ['edit_file', { path: 'a.txt', search_text: 'old', replacement_text: [] }],
    ['insert_text', { path: 'a.txt', line: 1, text: { value: 1 } }],
    ['insert_text', { path: 'a.txt', line: {}, text: 'text' }],
    ['replace_lines', { path: 'a.txt', start_line: [], end_line: 2, text: 'text' }],
    ['replace_lines', { path: 'a.txt', start_line: 1, end_line: 2, text: true }],
    ['batch_edit', { path: 'a.txt', edits: [null] }],
    ['batch_edit', { path: 'a.txt', edits: [{ search_text: 'old', replacement_text: {} }] }],
    ['run_command', { command: {} }],
    ['git', { command: [] }],
    ['spawn_process', { command: 42 }],
  ]

  it.each(malformed)('renders malformed %s as original JSON, without a diff', (name, args) => {
    for (const isError of [false, true]) {
      const html = body(name, args, isError)
      expect(html).toContain('args</span>')
      expect(html).toContain(isError ? 'Rejected: expected a string' : 'Result preserved')
      expect(html).not.toContain('>added</div>')
      expect(html).not.toContain('>removed</div>')
    }
  })

  it('preserves the original object content and executor rejection, not a fabricated edit', () => {
    const html = body('write_file', { path: 'answer.json', content: { count: 1000 } }, true)
    expect(html).toContain('&quot;content&quot;: {')
    expect(html).toContain('&quot;count&quot;: 1000')
    expect(html).toContain('>error</span>')
  })

  it('does not label a valid but failed write as added content', () => {
    const html = body('write_file', { path: 'a.txt', content: 'not written' }, true)
    expect(html).toContain('not written')
    expect(html).not.toContain('>added</div>')
  })

  it.each([
    ['write_file', { path: 'a.txt', content: '  indented\nnext' }],
    ['edit_file', { path: 'a.txt', search_text: 'old', replacement_text: 'next' }],
    ['insert_text', { path: 'a.txt', line: 1, text: 'next' }],
    ['replace_lines', { path: 'a.txt', start_line: 1, end_line: 2, text: 'next' }],
    ['batch_edit', { path: 'a.txt', edits: [{ search_text: 'old', replacement_text: 'next' }] }],
  ] as Array<[string, Record<string, unknown>]>)('keeps valid %s previews', (name, args) => {
    expect(body(name, args)).toContain('>added</div>')
  })

  it.each(['null', '[]', '[1]', '42', 'true', '"text"', '{broken'])('rejects a non-record argument envelope %s', value => {
    expect(parseToolArgs(value)).toBeNull()
  })

  it.each(['write_file', 'edit_file', 'list_directory', 'move_file', 'run_command', 'search_files', 'get_process_output'])(
    'keeps a malformed %s summary renderable', name => {
      const args = { path: {}, source: {}, command: {}, pattern: {}, pid: {}, content: {} }
      expect(typeof getToolSummary(name, args).context).toBe('string')
      const html = renderToStaticMarkup(React.createElement(InlineToolCall, {
        group: { name, statuses: [{ phase: 'calling', toolName: name, detail: JSON.stringify(args) }] },
        isStreaming: true,
      }))
      expect(html).toContain('data-vmlx-proof-tool-card="inline"')
      expect(html).not.toContain('[object Object]')
    },
  )

  it('does not mutate valid parsed arguments for the display summary', () => {
    const args = { path: 'a.txt', offset: 2, content: { nested: 'unchanged' } }
    const before = JSON.stringify(args)
    expect(getToolSummary('read_file', args).context).toBe('a.txt:2')
    expect(JSON.stringify(args)).toBe(before)
  })
})
