import { afterEach, describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $currentReasoningEffort, setCurrentReasoningEffort } from '@/store/session'
import { $sessionStates, publishSessionState } from '@/store/session-states'

import { applyRuntimeInfo } from './utils'

afterEach(() => {
  $activeSessionId.set(null)
  $sessionStates.set({})
  setCurrentReasoningEffort('')
  window.localStorage.clear()
})

describe('runtime effort and new-chat draft ownership', () => {
  it.each(['', 'medium', 'low'])('preserves draft %j while displaying the resumed session pin', draft => {
    setCurrentReasoningEffort(draft)
    const before = window.localStorage.getItem('hermes.desktop.composer.reasoning-effort')
    const patch = applyRuntimeInfo({ reasoning_effort: 'high', reasoning_effort_wire: 'high' })
    publishSessionState('resumed', { ...createClientSessionState('stored'), ...patch })
    $activeSessionId.set('resumed')
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('high')
    expect(window.localStorage.getItem('hermes.desktop.composer.reasoning-effort')).toBe(before)
    $activeSessionId.set(null)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe(draft)
    expect($currentReasoningEffort.get()).toBe(draft)
  })
})
