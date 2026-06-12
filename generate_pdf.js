const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--allow-file-access-from-files',
      '--disable-web-security'
    ]
  });

  const page = await browser.newPage();

  // A1 portrait @ 96 CSS dpi
  // 594mm × (96 / 25.4) = 2245px  /  841mm × (96 / 25.4) = 3179px
  const A1_W = 2245;
  const A1_H = 3179;

  // Design canvas: exactly the size .page uses in screen CSS
  const DESIGN_W = 900;
  const DESIGN_H = 1274;

  const scale = A1_W / DESIGN_W; // ≈ 2.4944

  // Viewport = A1 pixel dimensions → 1 CSS-px == 1 PDF-px at 96 dpi
  await page.setViewport({ width: A1_W, height: A1_H, deviceScaleFactor: 1 });

  const htmlPath =
    'C:/Users/ohjun/OneDrive/바탕 화면/Age-Detection-main/panel_final.html';
  await page.goto('file:///' + htmlPath, {
    waitUntil: 'networkidle0',
    timeout: 30000
  });

  // Wait for QR + fonts
  await new Promise(r => setTimeout(r, 2500));

  // Scale .page up to fill A1 and fix body to exact A1 dimensions
  await page.evaluate(
    (scale, w, h) => {
      const pageEl = document.querySelector('.page');

      // Remove screen-only decoration
      pageEl.style.borderRadius = '0';
      pageEl.style.boxShadow = 'none';

      // Position and scale
      pageEl.style.position = 'absolute';
      pageEl.style.top = '0';
      pageEl.style.left = '0';
      pageEl.style.transformOrigin = '0 0';
      pageEl.style.transform = `scale(${scale})`;

      // Body = exact A1 pixel canvas, no extra padding
      document.body.style.cssText = `
        margin: 0;
        padding: 0;
        width: ${w}px;
        height: ${h}px;
        overflow: hidden;
        background: #f4f6fb;
      `;
    },
    scale,
    A1_W,
    A1_H
  );

  const pdfPath =
    'C:/Users/ohjun/OneDrive/바탕 화면/Age-Detection-main/panel_final.pdf';

  // PDF page size matches A1: 594mm × 841mm = 2245px × 3179px @ 96dpi
  await page.pdf({
    path: pdfPath,
    width: '594mm',
    height: '841mm',
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }
  });

  await browser.close();
  console.log('완료:', pdfPath);
})();
