// Log-mel spectrogram in the browser, matching librosa:
//   librosa.feature.melspectrogram(y, sr=22050, n_fft=1024, hop_length=512, n_mels=128)  (center=True, zero pad, periodic Hann, power 2)
//   librosa.power_to_db(S, ref=np.max, top_db=80);  x = (dB + 40) / 20
// The mel filterbank itself is librosa's matrix, exported to mel_filters.json by train.py.
(function (root) {
  "use strict";

  function makeFFT(n) {
    const levels = Math.log2(n);
    if (!Number.isInteger(levels)) throw new Error("FFT size must be a power of two");
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < levels; b++) r |= ((i >>> b) & 1) << (levels - 1 - b);
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = -Math.sin(2 * Math.PI * i / n); }
    // In-place iterative radix-2 FFT (forward, e^{-i...}).
    return function fft(re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
      }
      for (let size = 2; size <= n; size *= 2) {
        const half = size / 2, step = n / size;
        for (let start = 0; start < n; start += size) {
          for (let k = 0; k < half; k++) {
            const a = start + k, b = a + half;
            const wr = cos[k * step], wi = sin[k * step];
            const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
            re[b] = re[a] - xr; im[b] = im[a] - xi;
            re[a] += xr; im[a] += xi;
          }
        }
      }
    };
  }

  /** Cut or zero-pad to cfg.n_samples. Longer recordings keep their loudest window (searched in hop steps). */
  function fitLength(y, cfg) {
    const n = cfg.n_samples;
    if (y.length === n) return Float32Array.from(y);
    if (y.length < n) { const out = new Float32Array(n); out.set(y); return out; }
    const csum = new Float64Array(y.length + 1);
    for (let i = 0; i < y.length; i++) csum[i + 1] = csum[i] + y[i] * y[i];
    let best = 0, bestE = -1;
    for (let s = 0; s + n <= y.length; s += cfg.hop) {
      const e = csum[s + n] - csum[s];
      if (e > bestE) { bestE = e; best = s; }
    }
    return y.slice(best, best + n);
  }

  /** y: Float32Array of exactly cfg.n_samples at cfg.sr. Returns Float32Array [n_mels * n_frames], row = mel bin. */
  function logMel(y, cfg) {
    const nfft = cfg.n_fft, hop = cfg.hop, nMels = cfg.n_mels, nFrames = cfg.n_frames, pad = nfft / 2;
    const fft = makeFFT(nfft);
    const win = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);   // periodic Hann
    const re = new Float64Array(nfft), im = new Float64Array(nfft), power = new Float64Array(nfft / 2 + 1);
    const mel = new Float64Array(nMels * nFrames);
    for (let t = 0; t < nFrames; t++) {
      const off = t * hop - pad;                       // center=True with zero padding
      for (let i = 0; i < nfft; i++) {
        const k = off + i;
        re[i] = (k >= 0 && k < y.length ? y[k] : 0) * win[i];
        im[i] = 0;
      }
      fft(re, im);
      for (let k = 0; k <= nfft / 2; k++) power[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < nMels; m++) {
        const f = cfg.filters[m], w = f.w;
        let s = 0;
        for (let j = 0; j < w.length; j++) s += power[f.start + j] * w[j];
        mel[m * nFrames + t] = s;
      }
    }
    // power_to_db(ref=max, amin, top_db) then the training normalisation
    let maxP = 0;
    for (let i = 0; i < mel.length; i++) if (mel[i] > maxP) maxP = mel[i];
    const refDb = 10 * Math.log10(Math.max(cfg.amin, maxP));
    let maxDb = -Infinity;
    for (let i = 0; i < mel.length; i++) {
      mel[i] = 10 * Math.log10(Math.max(cfg.amin, mel[i])) - refDb;
      if (mel[i] > maxDb) maxDb = mel[i];
    }
    const floor = maxDb - cfg.top_db, out = new Float32Array(mel.length);
    for (let i = 0; i < mel.length; i++) out[i] = (Math.max(mel[i], floor) + cfg.norm_offset) / cfg.norm_scale;
    return out;
  }

  const api = { makeFFT, fitLength, logMel };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LogMel = api;
})(typeof window !== "undefined" ? window : globalThis);
