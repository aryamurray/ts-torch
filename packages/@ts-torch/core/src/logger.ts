/**
 * Unified Logger for ts-torch
 *
 * Provides a single, configurable logging system for all ts-torch packages.
 * Supports log levels, custom handlers, and environment variable configuration.
 *
 * @example
 * ```ts
 * import { Logger } from '@ts-torch/core'
 *
 * // Set log level
 * Logger.setLevel('debug')
 *
 * // Log messages
 * Logger.info('Training started')
 * Logger.debug('Step 100: loss = 0.5')
 *
 * // Custom handler (e.g., file logging)
 * Logger.configure({
 *   handler: (level, message) => {
 *     fs.appendFileSync('training.log', `[${level}] ${message}\n`)
 *   }
 * })
 * ```
 */

/**
 * Log levels in order of verbosity (least to most)
 */
export type LogLevel = 'silent' | 'error' | 'warn' | 'info' | 'debug'

/**
 * Numeric values for log level comparison
 */
const LOG_LEVEL_VALUES: Record<LogLevel, number> = {
  silent: 0,
  error: 1,
  warn: 2,
  info: 3,
  debug: 4,
}

/**
 * Custom log handler function signature
 */
export type LogHandler = (level: LogLevel, message: string, ...args: unknown[]) => void

/**
 * Logger configuration options
 */
export interface LoggerConfig {
  /** Minimum log level to output (default: 'warn') */
  level?: LogLevel
  /** Prefix for all log messages (default: '[ts-torch]') */
  prefix?: string
  /** Custom handler function (replaces console output) */
  handler?: LogHandler
}

/**
 * Default console handler
 */
function defaultHandler(level: LogLevel, message: string, ...args: unknown[]): void {
  const formattedMessage = `[ts-torch] ${message}`

  switch (level) {
    case 'error':
      console.error(formattedMessage, ...args)
      break
    case 'warn':
      console.warn(formattedMessage, ...args)
      break
    case 'info':
    case 'debug':
      console.log(formattedMessage, ...args)
      break
  }
}

/**
 * Get initial log level from environment variables
 */
function getInitialLevel(): LogLevel {
  if (typeof process !== 'undefined' && process.env) {
    if (process.env.TS_TORCH_QUIET === '1') return 'silent'
    if (process.env.TS_TORCH_DEBUG === '1') return 'debug'
  }
  return 'warn'
}

/**
 * Internal logger state
 */
let currentLevel: LogLevel = getInitialLevel()
let currentHandler: LogHandler = defaultHandler

/**
 * Unified Logger for ts-torch
 *
 * All logging in ts-torch should go through this Logger to ensure
 * consistent behavior and allow users to control output.
 */
export const Logger = {
  /**
   * Configure the logger
   *
   * @param config - Configuration options
   *
   * @example
   * ```ts
   * Logger.configure({
   *   level: 'debug',
   *   prefix: '[my-app]',
   *   handler: (level, msg) => myLogger.log(level, msg)
   * })
   * ```
   */
  configure(config: LoggerConfig): void {
    if (config.level !== undefined) currentLevel = config.level
    // Note: prefix is ignored by default handler; use a custom handler to customize prefix
    if (config.handler !== undefined) currentHandler = config.handler
  },

  /**
   * Set the minimum log level
   *
   * @param level - Minimum level to output
   */
  setLevel(level: LogLevel): void {
    currentLevel = level
  },

  /**
   * Get the current log level
   */
  getLevel(): LogLevel {
    return currentLevel
  },

  /**
   * Check if a log level is enabled
   *
   * @param level - Level to check
   * @returns true if messages at this level will be output
   */
  isEnabled(level: LogLevel): boolean {
    return LOG_LEVEL_VALUES[level] <= LOG_LEVEL_VALUES[currentLevel]
  },

  /**
   * Log an error message
   *
   * @param message - Message to log
   * @param args - Additional arguments
   */
  error(message: string, ...args: unknown[]): void {
    if (this.isEnabled('error')) {
      currentHandler('error', message, ...args)
    }
  },

  /**
   * Log a warning message
   *
   * @param message - Message to log
   * @param args - Additional arguments
   */
  warn(message: string, ...args: unknown[]): void {
    if (this.isEnabled('warn')) {
      currentHandler('warn', message, ...args)
    }
  },

  /**
   * Log an info message
   *
   * @param message - Message to log
   * @param args - Additional arguments
   */
  info(message: string, ...args: unknown[]): void {
    if (this.isEnabled('info')) {
      currentHandler('info', message, ...args)
    }
  },

  /**
   * Log a debug message
   *
   * @param message - Message to log
   * @param args - Additional arguments
   */
  debug(message: string, ...args: unknown[]): void {
    if (this.isEnabled('debug')) {
      currentHandler('debug', message, ...args)
    }
  },

  /**
   * Reset logger to default configuration
   * Useful for testing
   */
  reset(): void {
    currentLevel = getInitialLevel()
    currentHandler = defaultHandler
  },

  /**
   * Create a child logger with a specific prefix
   * Useful for package-specific logging
   *
   * @param prefix - Prefix for this child logger
   * @returns Object with log methods using the prefix
   */
  child(prefix: string) {
    return {
      error: (message: string, ...args: unknown[]) => {
        if (Logger.isEnabled('error')) {
          currentHandler('error', `${prefix} ${message}`, ...args)
        }
      },
      warn: (message: string, ...args: unknown[]) => {
        if (Logger.isEnabled('warn')) {
          currentHandler('warn', `${prefix} ${message}`, ...args)
        }
      },
      info: (message: string, ...args: unknown[]) => {
        if (Logger.isEnabled('info')) {
          currentHandler('info', `${prefix} ${message}`, ...args)
        }
      },
      debug: (message: string, ...args: unknown[]) => {
        if (Logger.isEnabled('debug')) {
          currentHandler('debug', `${prefix} ${message}`, ...args)
        }
      },
    }
  },
}

/**
 * Map verbose number (0/1/2) to LogLevel
 * Used for backward compatibility with RL agent verbose config
 *
 * @param verbose - Numeric verbose level (0=warn, 1=info, 2=debug)
 * @returns Corresponding LogLevel
 */
export function verboseToLevel(verbose: number): LogLevel {
  if (verbose <= 0) return 'warn'
  if (verbose === 1) return 'info'
  return 'debug'
}
