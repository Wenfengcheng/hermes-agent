// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest'
import { captureFindScope, performScopedFind, releaseFindScope } from './find-in-page-scope'

afterEach(() => {
  releaseFindScope()
  document.body.innerHTML = ''
})

it('leaves the composer DOM and caret intact while searching and stepping', () => {
  document.body.innerHTML =
    '<div data-chat-surface><p>needle transcript</p><div contenteditable="true"><span>needle draft</span></div></div>'
  const surface = document.querySelector<HTMLElement>('[data-chat-surface]')!
  const composer = surface.querySelector<HTMLElement>('[contenteditable]')!
  const text = composer.querySelector('span')!.firstChild!
  const selection = window.getSelection()!
  selection.setPosition(text, 8)
  const before = composer.innerHTML
  for (const findNext of [false, true, false]) {
    const result = performScopedFind(surface, 'needle', {
      forward: true,
      findNext
    })
    expect(composer.innerHTML).toBe(before)
    expect(selection.anchorNode).toBe(text)
    expect(selection.anchorOffset).toBe(8)
    expect(result.count).toBe(1)
  }
})

it.each(['', 'true', 'TRUE', 'plaintext-only'])(
  'skips %s editors but still searches read-only transcript content',
  value => {
    document.body.innerHTML = `<div data-chat-surface><div contenteditable="${value}"><span>needle</span><span contenteditable="false">needle attachment</span></div><p contenteditable="false">needle transcript</p></div>`
    const surface = document.querySelector<HTMLElement>('[data-chat-surface]')!
    const editor = surface.firstElementChild!
    const original = editor.innerHTML
    expect(performScopedFind(surface, 'needle', { forward: true, findNext: false }).count).toBe(1)
    expect(editor.innerHTML).toBe(original)
    expect(performScopedFind(surface, '', { forward: true, findNext: false }).count).toBe(0)
    expect(editor.innerHTML).toBe(original)
  }
)

it('does not re-wrap transcript marks when a draft edit adds the active query', async () => {
  document.body.innerHTML =
    '<div data-chat-surface><p>needle transcript</p><div contenteditable="true">draft</div></div>'
  const surface = captureFindScope()!
  const editor = surface.querySelector<HTMLElement>('[contenteditable]')!
  performScopedFind(surface, 'needle', { forward: true, findNext: false })
  const mark = surface.querySelector('mark')!
  editor.appendChild(document.createTextNode(' needle'))
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(editor.querySelector('mark')).toBeNull()
  expect(surface.querySelector('mark')).toBe(mark)
  expect(performScopedFind(surface, 'needle', { forward: true, findNext: true }).count).toBe(1)
  expect(surface.querySelector('mark')).toBe(mark)
})
