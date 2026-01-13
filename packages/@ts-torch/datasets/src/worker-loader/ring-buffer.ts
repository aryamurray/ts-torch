/**
 * Ring buffer using SharedArrayBuffer for zero-copy data transfer between workers
 */

import type { BatchMetadata, BufferSlot, ReadySlot } from './types.js'
import { SlotState } from './types.js'

/**
 * Header layout in the SharedArrayBuffer:
 * - First N * 4 bytes: Int32Array for slot states (one int32 per slot)
 * - Remaining bytes: Data slots
 */
const BYTES_PER_STATE = 4 // Int32

/**
 * Type declaration for Atomics.waitAsync (ES2024)
 * Needed because TypeScript's lib may not include this yet
 */
declare global {
  interface Atomics {
    waitAsync(
      typedArray: Int32Array,
      index: number,
      value: number,
      timeout?: number,
    ): { async: false; value: 'ok' | 'not-equal' | 'timed-out' } | { async: true; value: Promise<'ok' | 'timed-out'> }
  }
}

/**
 * Ring buffer for worker data loading
 *
 * Uses SharedArrayBuffer + Atomics for lock-free synchronization between
 * the main thread and worker threads.
 */
export class RingBuffer {
  private buffer: SharedArrayBuffer
  private stateArray: Int32Array
  private slots: BufferSlot[]
  private metadata: Map<number, BatchMetadata> = new Map()
  private numSlots: number
  private headerSize: number

  constructor(numSlots: number, bytesPerSlot: number) {
    this.numSlots = numSlots

    // Header: state array (one int32 per slot)
    this.headerSize = numSlots * BYTES_PER_STATE

    // Total size: header + all slots
    const totalBytes = this.headerSize + numSlots * bytesPerSlot

    this.buffer = new SharedArrayBuffer(totalBytes)
    this.stateArray = new Int32Array(this.buffer, 0, numSlots)

    // Initialize slots
    this.slots = []
    for (let i = 0; i < numSlots; i++) {
      this.slots.push({
        index: i,
        byteOffset: this.headerSize + i * bytesPerSlot,
        maxBytes: bytesPerSlot,
      })
      // Initialize state to EMPTY
      Atomics.store(this.stateArray, i, SlotState.EMPTY)
    }
  }

  /**
   * Get the underlying SharedArrayBuffer
   */
  getBuffer(): SharedArrayBuffer {
    return this.buffer
  }

  /**
   * Get slot info by index
   */
  getSlot(index: number): BufferSlot {
    const slot = this.slots[index]
    if (!slot) {
      throw new Error(`Invalid slot index: ${index}`)
    }
    return slot
  }

  /**
   * Try to acquire an empty slot for writing.
   * Returns the slot if successful, null if no slots available.
   */
  tryAcquireWriteSlot(): BufferSlot | null {
    for (let i = 0; i < this.numSlots; i++) {
      // Try to atomically change state from EMPTY to LOADING
      const oldState = Atomics.compareExchange(
        this.stateArray,
        i,
        SlotState.EMPTY,
        SlotState.LOADING,
      )

      if (oldState === SlotState.EMPTY) {
        // Successfully acquired slot
        return this.slots[i]!
      }
    }

    return null // No slots available
  }

  /**
   * Acquire an empty slot for writing, waiting if necessary.
   */
  async acquireWriteSlot(): Promise<BufferSlot> {
    while (true) {
      const slot = this.tryAcquireWriteSlot()
      if (slot) {
        return slot
      }

      // No slots available, wait a bit and retry
      // In a real implementation we'd use Atomics.waitAsync on a condition,
      // but for simplicity we poll with a short delay
      await new Promise((resolve) => setTimeout(resolve, 1))
    }
  }

  /**
   * Mark a slot as ready after a worker has finished loading data.
   * Called from the main thread after receiving 'batchReady' message.
   */
  markReady(slotIndex: number, metadata: BatchMetadata): void {
    this.metadata.set(slotIndex, metadata)
    Atomics.store(this.stateArray, slotIndex, SlotState.READY)
    // Wake up any waiters
    Atomics.notify(this.stateArray, slotIndex)
  }

  /**
   * Wait for the next ready slot in order.
   * The readIndex parameter indicates which slot we're waiting for.
   */
  async waitForReady(readIndex: number): Promise<ReadySlot> {
    const slotIndex = readIndex % this.numSlots

    while (true) {
      const state = Atomics.load(this.stateArray, slotIndex)

      if (state === SlotState.READY) {
        // Mark as consuming
        Atomics.store(this.stateArray, slotIndex, SlotState.CONSUMING)

        const metadata = this.metadata.get(slotIndex)
        if (!metadata) {
          throw new Error(`No metadata for slot ${slotIndex}`)
        }

        return {
          ...this.slots[slotIndex]!,
          buffer: this.buffer,
          metadata,
        }
      }

      // Wait for state change
      // Atomics.waitAsync returns { async: true, value: Promise } or { async: false, value: string }
      const waitResult = Atomics.waitAsync(this.stateArray, slotIndex, state, 1000)

      if (waitResult.async) {
        await waitResult.value
      }
    }
  }

  /**
   * Release a slot after the batch has been consumed.
   */
  release(slotIndex: number): void {
    this.metadata.delete(slotIndex)
    Atomics.store(this.stateArray, slotIndex, SlotState.EMPTY)
    // Wake up any writers waiting for empty slots
    Atomics.notify(this.stateArray, slotIndex)
  }

  /**
   * Reset all slots to empty state.
   */
  reset(): void {
    this.metadata.clear()
    for (let i = 0; i < this.numSlots; i++) {
      Atomics.store(this.stateArray, i, SlotState.EMPTY)
    }
  }

  /**
   * Get current state of a slot (for debugging)
   */
  getSlotState(slotIndex: number): SlotState {
    return Atomics.load(this.stateArray, slotIndex)
  }

  /**
   * Get number of slots
   */
  get length(): number {
    return this.numSlots
  }

  /**
   * Check if slot is in LOADING state (a worker is writing to it)
   */
  isLoading(slotIndex: number): boolean {
    return Atomics.load(this.stateArray, slotIndex) === SlotState.LOADING
  }
}
