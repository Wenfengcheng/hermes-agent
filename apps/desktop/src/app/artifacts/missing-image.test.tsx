import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { ArtifactsView } from './index'

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({ sessions: [{ id: 'fixture', title: 'Fixture' }] }),
  getSessionMessages: async () => ({
    messages: [{ role: 'assistant', timestamp: 1000, content: 'MEDIA:/tmp/generated/image.png' }]
  })
}))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function gallery() {
  return render(<MemoryRouter><ArtifactsView /></MemoryRouter>)
}

it('explains a missing image without removing its path or chat action', async () => {
  const readFileDataUrl = vi.fn().mockRejectedValue(new Error('ENOENT'))
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  gallery()

  await screen.findByText('Preview unavailable')
  expect(readFileDataUrl).toHaveBeenCalledWith('/tmp/generated/image.png')
  expect(screen.getByText('/tmp/generated/image.png')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Chat' })).toBeTruthy()
  expect(screen.queryByRole('img', { name: 'image.png' })).toBeNull()
})

it('replaces an undecodable image with the unavailable preview state', async () => {
  vi.stubGlobal('hermesDesktop', { readFileDataUrl: vi.fn().mockResolvedValue('data:image/png;base64,INVALID') })
  gallery()

  fireEvent.error(await screen.findByRole('img', { name: 'image.png' }))
  await screen.findByText('Preview unavailable')
  expect(screen.queryByRole('img', { name: 'image.png' })).toBeNull()
})

it('keeps successful previews unchanged', async () => {
  const src = 'data:image/png;base64,TE9DQUw='
  vi.stubGlobal('hermesDesktop', { readFileDataUrl: vi.fn().mockResolvedValue(src) })
  gallery()

  expect((await screen.findByRole('img', { name: 'image.png' })).getAttribute('src')).toBe(src)
  expect(screen.queryByText('Preview unavailable')).toBeNull()
})

it('does not call a still-pending preview unavailable', async () => {
  const readFileDataUrl = vi.fn(() => new Promise<string>(() => {}))
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })
  gallery()

  await waitFor(() => expect(readFileDataUrl).toHaveBeenCalled())
  expect(screen.queryByText('Preview unavailable')).toBeNull()
})
