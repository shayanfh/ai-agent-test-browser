import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'VoiceBench — Local voice latency lab', description: 'Measure Moonshine, Gemma and Piper end-to-end voice latency on your own server.' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
