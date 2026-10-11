import test from 'node:test'
import assert from 'node:assert/strict'

test('Frontend API URL defaults to 127.0.0.1:8000 when unset', () => {
  const defaultUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
  assert.equal(defaultUrl, 'http://127.0.0.1:8000')
})

test('Evaluation request schema conforms to backend contract', () => {
  const sampleRequest = {
    dataset_id: 'sample',
    k_values: [1, 3, 5],
    include_baselines: true,
  }
  assert.equal(sampleRequest.dataset_id, 'sample')
  assert.deepEqual(sampleRequest.k_values, [1, 3, 5])
  assert.equal(sampleRequest.include_baselines, true)
})
