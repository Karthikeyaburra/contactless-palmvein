/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neo: {
          bg: '#FFFDF0',
          black: '#121212',
          yellow: '#FFDE59',
          cyan: '#38BDF8',
          neon: '#00F0FF',
          pink: '#FF4081',
          coral: '#FF3366',
          orange: '#FF7A00',
          purple: '#A855F7',
          lime: '#CCFF00',
          card: '#FFFFFF',
          muted: '#8C919B',
          light: '#F4F4F0',
        },
      },
      boxShadow: {
        'neo': '4px 4px 0px #121212',
        'neo-sm': '2px 2px 0px #121212',
        'neo-lg': '6px 6px 0px #121212',
        'neo-xl': '8px 8px 0px #121212',
        'neo-active': '1px 1px 0px #121212',
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'sans-serif'],
        display: ['"Space Grotesk"', '"Plus Jakarta Sans"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
