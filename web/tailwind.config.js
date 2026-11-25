/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // every colour resolves to the variables in styles.css, so the palette block
        // stays the single place a colour is chosen
        brand: {
          DEFAULT: 'var(--polyu-red)',
          dark: 'var(--polyu-red-dark)',
          tint: 'var(--polyu-red-tint)',
        },
        ink: 'var(--text)',
        muted: 'var(--muted)',
        panel: {
          DEFAULT: 'var(--panel)',
          soft: 'var(--panel-2)',
          glass: 'var(--panel-glass)',
        },
        line: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        onaccent: {
          DEFAULT: 'var(--on-accent)',
          dim: 'var(--on-accent-dim)',
        },
        accent: {
          soft: 'var(--accent-soft)',
          ring: 'var(--accent-ring)',
        },
        ok: 'var(--ok)',
        warn: 'var(--warn)',
        danger: {
          DEFAULT: 'var(--danger)',
          soft: 'var(--danger-soft)',
        },
      },
      borderRadius: {
        panel: 'var(--radius)',
      },
      fontFamily: {
        sans: ['var(--font)'],
      },
    },
  },
  plugins: [],
};
