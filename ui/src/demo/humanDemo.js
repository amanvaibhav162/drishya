/**
 * DRISHYA Realistic Human-Paced Interactive Demonstration Engine
 * 
 * Authentic Windows Cursor System:
 * - Dynamically morphs cursor based on interactive context:
 *   • 'default': Classic Windows white arrow with black outline
 *   • 'pointer': Windows pointing hand over buttons, tabs, links, and select dropdowns
 *   • 'text': Windows I-beam text selector over input fields and search boxes
 *   • 'ew-resize': Windows horizontal double arrow (<->) over range sliders and opacity wheels
 * 
 * Natural Kinematics & Scrolling:
 * - Organic mouse wheel scrolling in progressive human-paced impulses with physical inertia
 * - Synchronous React controlled input updates for the sideways opacity slider
 * - Strict modal containment: Never scrolls background while modal is open; closes modal cleanly
 * - Shows all 4 tabs in Judge Inspector with live opacity slider blending
 */

let activeCursor = null;
let currentCursorType = 'default';
let currentHotspotX = 320;
let currentHotspotY = 180;

const CURSOR_HOTSPOTS = {
  default: { x: 1, y: 1 },
  pointer: { x: 6, y: 1 },
  text: { x: 8, y: 11 },
  'ew-resize': { x: 12, y: 8 }
};

const CURSOR_SVGS = {
  default: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.38));">
      <path d="M1 1 L1 18 L5.5 13.5 L9.2 21.2 L12.4 19.8 L8.8 12.5 L14.5 12.5 Z" fill="#FFFFFF" stroke="#111111" stroke-width="1.3" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
  `,
  pointer: `
    <svg width="22" height="24" viewBox="0 0 22 24" fill="none" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.38));">
      <path d="M6 1 C5.4 1 5 1.4 5 2 L5 11 L4.2 10.2 C3.7 9.7 2.9 9.7 2.4 10.2 C1.9 10.7 1.9 11.5 2.4 12 L7.5 17.1 C8.5 18.1 9.8 18.7 11.2 18.7 L14.5 18.7 C16.4 18.7 18 17.1 18 15.2 L18 11 C18 10.4 17.6 10 17 10 C16.4 10 16 10.4 16 11 L16 10 C16 9.4 15.6 9 15 9 C14.4 9 14 9.4 14 10 L14 9 C14 8.4 13.6 8 13 8 C12.4 8 12 8.4 12 9 L12 2 C12 1.4 11.6 1 11 1 C10.4 1 10 1.4 10 2 L10 10 L6 10 L6 2 C6 1.4 5.6 1 5 1 Z" fill="#FFFFFF" stroke="#111111" stroke-width="1.2" stroke-linejoin="round"/>
    </svg>
  `,
  text: `
    <svg width="18" height="22" viewBox="0 0 16 22" fill="none" style="filter: drop-shadow(0 1px 3px rgba(0,0,0,0.45));">
      <path d="M2.5 1 H13.5 M8 1 V21 M2.5 21 H13.5" stroke="#111111" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M2.5 1 H13.5 M8 1 V21 M2.5 21 H13.5" stroke="#FFFFFF" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `,
  'ew-resize': `
    <svg width="24" height="16" viewBox="0 0 24 16" fill="none" style="filter: drop-shadow(0 1px 3px rgba(0,0,0,0.4));">
      <path d="M3 8 L7.5 3.5 L7.5 6.2 L16.5 6.2 L16.5 3.5 L21 8 L16.5 12.5 L16.5 9.8 L7.5 9.8 L7.5 12.5 Z" fill="#FFFFFF" stroke="#111111" stroke-width="1.2" stroke-linejoin="round"/>
    </svg>
  `
};

function getOrCreateCursor() {
  if (activeCursor && document.body.contains(activeCursor)) {
    activeCursor.style.display = 'block';
    return activeCursor;
  }

  let cursor = document.getElementById('demo-virtual-cursor');
  if (!cursor) {
    cursor = document.createElement('div');
    cursor.id = 'demo-virtual-cursor';
    cursor.innerHTML = CURSOR_SVGS.default;
    cursor.style.cssText = `
      position: fixed;
      top: 179px;
      left: 319px;
      width: 24px;
      height: 24px;
      z-index: 99999999;
      pointer-events: none;
      transform-origin: 0 0;
      transition: transform 0.12s ease-out;
      will-change: transform, left, top;
    `;
    document.body.appendChild(cursor);
  }
  cursor.style.display = 'block';
  activeCursor = cursor;
  return cursor;
}

function updateCursorPosition(hotspotX, hotspotY) {
  currentHotspotX = hotspotX;
  currentHotspotY = hotspotY;
  const cursor = getOrCreateCursor();
  const hs = CURSOR_HOTSPOTS[currentCursorType] || { x: 0, y: 0 };
  cursor.style.left = `${(hotspotX - hs.x).toFixed(1)}px`;
  cursor.style.top = `${(hotspotY - hs.y).toFixed(1)}px`;
}

function setCursorType(type = 'default') {
  if (currentCursorType === type) return;
  const cursor = getOrCreateCursor();
  if (CURSOR_SVGS[type]) {
    cursor.innerHTML = CURSOR_SVGS[type];
    currentCursorType = type;
    updateCursorPosition(currentHotspotX, currentHotspotY);
  }
}

function removeCursor() {
  const cursor = document.getElementById('demo-virtual-cursor');
  if (cursor) {
    cursor.style.display = 'none';
  }
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Returns the contextual Windows cursor type for a given DOM element
 */
function getCursorTypeForElement(el) {
  if (!el) return 'default';
  const tag = el.tagName?.toLowerCase();
  const type = el.getAttribute('type')?.toLowerCase();
  const role = el.getAttribute('role');

  if (tag === 'input' && (type === 'range' || el.id === 'slider-heatmap-opacity')) {
    return 'ew-resize';
  }
  if (tag === 'input' && (type === 'text' || type === 'number' || type === 'tel' || !type)) {
    return 'text';
  }
  if (tag === 'textarea') {
    return 'text';
  }
  if (
    tag === 'button' ||
    tag === 'a' ||
    tag === 'select' ||
    role === 'button' ||
    el.classList?.contains('sample-scan-pill') ||
    el.classList?.contains('nav-link-exact') ||
    el.classList?.contains('tab-btn') ||
    el.classList?.contains('btn') ||
    el.classList?.contains('eye-selector-badge') ||
    el.closest('button')
  ) {
    return 'pointer';
  }
  return 'default';
}

/**
 * Returns the actual active scrollable container in DRISHYA (.main-content).
 */
function getScrollContainer() {
  const main = document.querySelector('.main-content');
  if (main && main.scrollHeight > main.clientHeight) {
    return main;
  }
  return document.scrollingElement || document.documentElement || document.body || window;
}

function getScrollTop() {
  const sc = getScrollContainer();
  return sc === window ? (window.pageYOffset || document.documentElement.scrollTop || 0) : sc.scrollTop;
}

/**
 * Organic mouse wheel scrolling in progressive impulses with inertia
 */
async function humanWheelScroll(deltaY, steps = 12, delayMs = 36) {
  const sc = getScrollContainer();
  const stepAmount = deltaY / steps;

  for (let i = 0; i < steps; i++) {
    const p = (i + 0.5) / steps;
    const impulse = Math.sin(p * Math.PI) * (1.35 - p * 0.4);
    const scrollInc = stepAmount * impulse * 1.35;

    if (sc === window) {
      window.scrollBy({ top: scrollInc, behavior: 'instant' });
    } else {
      sc.scrollTop += scrollInc;
    }
    await sleep(delayMs);
  }
  await sleep(220);
}

/**
 * Smooth natural wheel scroll back to top of container
 */
async function humanWheelScrollToTop() {
  const sc = getScrollContainer();
  while (getScrollTop() > 30) {
    const remaining = getScrollTop();
    const jump = Math.min(remaining, Math.max(160, remaining * 0.42));
    if (sc === window) {
      window.scrollBy({ top: -jump, behavior: 'instant' });
    } else {
      sc.scrollTop -= jump;
    }
    await sleep(35);
  }
  if (sc === window) window.scrollTo(0, 0);
  else sc.scrollTop = 0;
  await sleep(350);
}

/**
 * Asymmetric human velocity warp based on Flash & Hogan minimum jerk model:
 * Rapid ballistic acceleration peaking early (~32%), followed by a gentle,
 * prolonged deceleration and visual landing tail.
 */
function humanVelocityProfile(p) {
  const peak = 0.32;
  let warped;
  if (p <= peak) {
    warped = 0.5 * Math.pow(p / peak, 1.42);
  } else {
    warped = 0.5 + 0.5 * (1 - Math.pow((1 - p) / (1 - peak), 2.15));
  }
  return 10 * Math.pow(warped, 3) - 15 * Math.pow(warped, 4) + 6 * Math.pow(warped, 5);
}

/**
 * Automatically and naturally scrolls the container with progressive human mouse wheel impulses
 * until the target element is safely and comfortably positioned in the viewport (between 25% and 75% height).
 */
async function ensureElementInView(el) {
  if (!el) return;
  // If element is inside a modal or fixed overlay, do not scroll the background page
  if (el.closest('.modal-overlay') || el.closest('.modal-card')) {
    return;
  }

  const vh = window.innerHeight;
  let rect = el.getBoundingClientRect();

  // If element is not comfortably in view (margins: 90px top, 90px bottom)
  if (rect.top < 90 || rect.bottom > vh - 90) {
    const targetY = vh * 0.48; // Bring element to comfortable vertical center
    const currentCenterY = rect.top + rect.height / 2;
    const deltaY = currentCenterY - targetY;

    if (Math.abs(deltaY) > 20) {
      const steps = Math.max(8, Math.min(16, Math.round(Math.abs(deltaY) / 32)));
      await humanWheelScroll(deltaY, steps, 30);
      await sleep(220);
    }
  }
}

/**
 * Moves Windows cursor with authentic human kinematics:
 * 1. Automatic viewport awareness: scrolls off-screen elements into view before computing coordinates
 * 2. Forearm + wrist decoupled Cubic Bezier trajectory (varying curve arcs and occasional S-curves)
 * 3. Asymmetric Flash-Hogan minimum jerk velocity (fast initial sweep + long deceleration)
 * 4. Human motor imperfection: subtle overshoot & corrective settling on fast sweeps
 * 5. Continuous multi-frequency harmonic hand drift (~7 Hz & ~11 Hz biological tremor)
 * 6. Pre-movement eye-fixation hesitation & post-arrival dwell
 * 7. Dynamic collision-based cursor morphing matching desktop OS behavior
 * 8. Natural landing variance (humans never click dead center)
 */
async function moveCursorTo(elOrSelector, offsetX = 0, offsetY = 0) {
  const el = typeof elOrSelector === 'string' ? document.querySelector(elOrSelector) : elOrSelector;
  if (!el) return null;

  // 1. GUARANTEE element is comfortably visible in the viewport BEFORE moving cursor
  await ensureElementInView(el);

  const vh = window.innerHeight;
  const vw = window.innerWidth;
  const targetCursorType = getCursorTypeForElement(el);

  // Fresh bounding rect after smooth scroll has settled
  let rect = el.getBoundingClientRect();
  const startX = currentHotspotX;
  const startY = currentHotspotY;

  // Natural human landing variance: slight organic offset within target area
  let rawTargetX, rawTargetY;
  if (targetCursorType === 'text') {
    rawTargetX = rect.left + 16 + (Math.random() * 16) + offsetX;
    rawTargetY = rect.top + rect.height / 2 + (Math.random() - 0.5) * 4 + offsetY;
  } else if (targetCursorType === 'pointer') {
    rawTargetX = rect.left + rect.width * (0.35 + Math.random() * 0.3) + offsetX;
    rawTargetY = rect.top + rect.height * (0.38 + Math.random() * 0.24) + offsetY;
  } else if (targetCursorType === 'ew-resize') {
    rawTargetX = rect.left + rect.width / 2 + offsetX;
    rawTargetY = rect.top + rect.height / 2 + offsetY;
  } else {
    rawTargetX = rect.left + rect.width * (0.35 + Math.random() * 0.3) + offsetX;
    rawTargetY = rect.top + rect.height * (0.35 + Math.random() * 0.3) + offsetY;
  }

  const destX = Math.round(Math.min(vw - 25, Math.max(15, rawTargetX)));
  const destY = Math.round(Math.min(vh - 25, Math.max(45, rawTargetY)));

  const dx = destX - startX;
  const dy = destY - startY;
  const dist = Math.sqrt(dx * dx + dy * dy);

  if (dist < 6) {
    setCursorType(targetCursorType);
    await sleep(80);
    return el;
  }

  // 1. Natural pre-movement eye-fixation delay (human visual acquisition)
  if (dist > 35) {
    await sleep(80 + Math.floor(Math.random() * 110));
  }
  // Subtle pre-motion finger twitch before large sweeps
  if (dist > 220) {
    const twitchX = (Math.random() - 0.5) * 1.8;
    const twitchY = (Math.random() - 0.5) * 1.8;
    updateCursorPosition(startX + twitchX, startY + twitchY);
    await sleep(35);
  }

  // 2. Fitts's Law duration scaling (680ms to 1300ms)
  const duration = Math.max(680, Math.min(1300, Math.round(520 + Math.sqrt(dist) * 36 + dist * 0.35 + Math.random() * 110)));

  // 3. Forearm and wrist decoupled Cubic Bezier control points
  const nX = -dy / dist;
  const nY = dx / dist;

  let bowDir = (dx * dy > 0) ? -1 : 1;
  if (Math.random() < 0.28) bowDir *= -1;

  const baseBow = Math.min(44, Math.max(8, dist * 0.13));
  const bow1 = bowDir * (baseBow * (0.8 + Math.random() * 0.4));

  // 30% chance of S-curve (forearm sweeps outward, wrist counter-aligns)
  const isSCurve = dist > 130 && Math.random() < 0.32;
  const bow2 = isSCurve ? -bow1 * (0.45 + Math.random() * 0.35) : bow1 * (0.5 + Math.random() * 0.5);

  const t1 = 0.28 + (Math.random() - 0.5) * 0.08;
  const t2 = 0.72 + (Math.random() - 0.5) * 0.08;

  const ctrl1X = startX + dx * t1 + nX * bow1;
  const ctrl1Y = startY + dy * t1 + nY * bow1;
  const ctrl2X = startX + dx * t2 + nX * bow2;
  const ctrl2Y = startY + dy * t2 + nY * bow2;

  // 4. Overshoot dynamics: ~46% of medium-to-long sweeps slightly overshoot and pull back
  const hasOvershoot = dist > 85 && Math.random() < 0.46;
  const overshootDist = hasOvershoot ? Math.min(9, Math.max(3.5, dist * 0.03 + Math.random() * 3.5)) : 0;
  const overX = (dx / dist) * overshootDist + nX * (Math.random() - 0.5) * 2;
  const overY = (dy / dist) * overshootDist + nY * (Math.random() - 0.5) * 2;

  // 5. Continuous biological tremor phase
  const phase1 = Math.random() * Math.PI * 2;
  const phase2 = Math.random() * Math.PI * 2;

  const cursor = getOrCreateCursor();
  cursor.style.transition = 'none';

  const startTime = performance.now();
  let iconMorphed = false;

  await new Promise((resolve) => {
    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Asymmetric velocity profile (Meyer's two-component motor model)
      const u = humanVelocityProfile(progress);

      // Cubic Bezier interpolation
      const invU = 1 - u;
      let curX = invU * invU * invU * startX +
                 3 * invU * invU * u * ctrl1X +
                 3 * invU * u * u * ctrl2X +
                 u * u * u * destX;
      let curY = invU * invU * invU * startY +
                 3 * invU * invU * u * ctrl1Y +
                 3 * invU * u * u * ctrl2Y +
                 u * u * u * destY;

      // Realistic Overshoot impulse: peaks at ~88% progress, smoothly settles back
      if (hasOvershoot && progress > 0.72) {
        const overP = (progress - 0.72) / 0.28;
        const overAmp = Math.sin(overP * Math.PI);
        curX += overX * overAmp;
        curY += overY * overAmp;
      }

      // Smooth harmonic motor sway (~7 Hz & ~11.5 Hz postural tremor)
      if (progress > 0.05 && progress < 0.95) {
        const swayAmp = Math.sin(progress * Math.PI);
        const swayX = Math.sin(elapsed * 0.0072 + phase1) * 0.85 + Math.sin(elapsed * 0.015) * 0.35;
        const swayY = Math.cos(elapsed * 0.0065 + phase2) * 0.75 + Math.cos(elapsed * 0.013) * 0.25;
        curX += swayX * swayAmp;
        curY += swayY * swayAmp;
      }

      updateCursorPosition(curX, curY);

      // Collision-based cursor morphing: morphs when cursor enters target bounds
      if (!iconMorphed) {
        const b = el.getBoundingClientRect();
        const inside = (
          curX >= b.left - 6 &&
          curX <= b.right + 6 &&
          curY >= b.top - 6 &&
          curY <= b.bottom + 6
        );
        if (inside || progress > 0.85) {
          setCursorType(targetCursorType);
          iconMorphed = true;
        } else if (progress < 0.65 && currentCursorType !== 'default') {
          setCursorType('default');
        }
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        updateCursorPosition(destX, destY);
        setCursorType(targetCursorType);
        resolve();
      }
    }
    requestAnimationFrame(step);
  });

  // Post-arrival dwell (human visual confirmation before committing next action)
  await sleep(180 + Math.floor(Math.random() * 110));
  return el;
}

/**
 * Realistic mouse click with downstroke and Windows blue accent pulse
 */
async function humanClick(elOrSelector, offsetX = 0, offsetY = 0) {
  const el = await moveCursorTo(elOrSelector, offsetX, offsetY);
  if (!el) return;

  const cursor = getOrCreateCursor();

  // Natural human pause before pressing down
  await sleep(110 + Math.floor(Math.random() * 70));

  // Mouse button down
  cursor.style.transform = 'scale(0.86)';
  await sleep(85);

  // Click pulse emanating from hotspot
  const hs = CURSOR_HOTSPOTS[currentCursorType] || { x: 0, y: 0 };
  const ripple = document.createElement('div');
  ripple.style.cssText = `
    position: absolute;
    top: ${hs.y - 11}px;
    left: ${hs.x - 11}px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(37, 99, 235, 0.28);
    border: 1.8px solid #2563EB;
    transform: scale(0.4);
    opacity: 1;
    transition: transform 0.35s cubic-bezier(0.1, 0.9, 0.2, 1), opacity 0.35s ease-out;
    pointer-events: none;
  `;
  cursor.appendChild(ripple);
  requestAnimationFrame(() => {
    ripple.style.transform = 'scale(2.2)';
    ripple.style.opacity = '0';
  });
  setTimeout(() => ripple.remove(), 380);

  // Mouse button release
  cursor.style.transform = 'scale(1)';
  await sleep(75);

  el.click();

  if (typeof el.blur === 'function' && el.tagName === 'BUTTON') {
    el.blur();
  }

  // Visual confirmation pause after clicking
  await sleep(420 + Math.floor(Math.random() * 120));
}

/**
 * Human typing cadence with authentic I-beam cursor and caret simulation
 */
async function humanType(elOrSelector, text, baseDelay = 180) {
  const el = typeof elOrSelector === 'string' ? document.querySelector(elOrSelector) : elOrSelector;
  if (!el) return;

  await humanClick(el);
  setCursorType('text');
  el.focus();
  await sleep(220);

  el.value = '';
  const tracker = el._valueTracker;
  if (tracker) tracker.setValue('dummy');

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    el.value = text.slice(0, i + 1);
    if (tracker) tracker.setValue(text.slice(0, i));

    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));

    let delay = baseDelay + (Math.random() * 50 - 25);
    if (char === ' ') {
      delay += 240 + Math.random() * 90;
    } else if (char === '-' || char === '+' || char === '(' || char === ')') {
      delay += 120;
    } else if (!isNaN(char)) {
      delay += 35;
    }

    await sleep(Math.max(85, Math.round(delay)));
  }

  await sleep(400 + Math.floor(Math.random() * 120));
  setCursorType('default');
}

async function humanSelect(selectSelector, value) {
  const select = typeof selectSelector === 'string' ? document.querySelector(selectSelector) : selectSelector;
  if (!select) return;

  setCursorType('pointer');
  await humanClick(select);
  select.focus();
  select.value = value;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  select.dispatchEvent(new Event('change', { bubbles: true }));
  await sleep(550);
  setCursorType('default');
}

/**
 * Drags the sideways scroll wheel / range slider with synchronous React controlled input state updates
 */
async function humanDragSlider(sliderSelector, targetValue, durationMs = 950) {
  const slider = await moveCursorTo(sliderSelector);
  if (!slider) return;

  setCursorType('ew-resize');
  await sleep(220);

  const startVal = Number(slider.value || 40);
  const diff = targetValue - startVal;
  const rect = slider.getBoundingClientRect();
  const startX = rect.left + (rect.width * (startVal / 100));
  const endX = rect.left + (rect.width * (targetValue / 100));
  const sliderY = rect.top + rect.height / 2;

  updateCursorPosition(startX, sliderY);
  await sleep(280);

  // Grip down
  const cursor = getOrCreateCursor();
  cursor.style.transform = 'scale(0.88)';
  await sleep(110);

  const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;

  // Real human dragging: slight overshoot and corrective settlement
  const hasSliderOvershoot = Math.abs(diff) > 20;
  const sliderOvershoot = hasSliderOvershoot ? (diff > 0 ? 2 : -2) : 0;
  const peakVal = targetValue + sliderOvershoot;

  const startTime = performance.now();
  await new Promise((resolve) => {
    function step(now) {
      const elapsed = now - startTime;
      const p = Math.min(elapsed / durationMs, 1);

      const ease = p < 0.4
        ? Math.pow(p / 0.4, 1.3) * 0.4
        : 0.4 + (1 - Math.pow((1 - p) / 0.6, 1.8)) * 0.6;

      let curVal;
      if (hasSliderOvershoot && p > 0.75) {
        const backP = (p - 0.75) / 0.25;
        curVal = Math.round(peakVal - sliderOvershoot * backP);
      } else {
        curVal = Math.round(startVal + (peakVal - startVal) * ease);
      }
      curVal = Math.max(0, Math.min(100, curVal));

      const curX = rect.left + (rect.width * (curVal / 100));

      if (valueSetter) {
        valueSetter.call(slider, curVal);
      } else {
        slider.value = curVal;
      }
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      slider.dispatchEvent(new Event('change', { bubbles: true }));

      updateCursorPosition(curX, sliderY);

      if (p < 1) {
        requestAnimationFrame(step);
      } else {
        if (valueSetter) valueSetter.call(slider, targetValue);
        else slider.value = targetValue;
        slider.dispatchEvent(new Event('input', { bubbles: true }));
        slider.dispatchEvent(new Event('change', { bubbles: true }));
        updateCursorPosition(endX, sliderY);
        resolve();
      }
    }
    requestAnimationFrame(step);
  });

  cursor.style.transform = 'scale(1)';
  await sleep(380);
  setCursorType('default');
}

/**
 * Helper to wait until AI screening analysis completes
 */
async function waitForScreeningToFinish(timeoutMs = 18000) {
  const startTime = Date.now();
  await sleep(700);
  while (Date.now() - startTime < timeoutMs) {
    const isProcessing = document.getElementById('screening-progress-card') || 
                         document.querySelector('.btn-portal-process[disabled]');
    const hasResult = document.querySelector('.card-step[style*="border: 1.5px solid"]');
    if (!isProcessing && hasResult) {
      await sleep(600);
      return true;
    }
    await sleep(400);
  }
  return false;
}

/**
 * Helper to wait until sample image is loaded into DOM
 */
async function waitForImageLoaded(timeoutMs = 6000) {
  const startTime = Date.now();
  await sleep(400);
  while (Date.now() - startTime < timeoutMs) {
    if (document.querySelector('.retina-img') || document.querySelector('.image-canvas-wrapper img')) {
      await sleep(600);
      return true;
    }
    await sleep(300);
  }
  return false;
}

/**
 * Comprehensive, natural showcase for each DR image:
 * 1. Progressive human mouse wheel scroll down through retina photo & green IQA badge
 * 2. Progressive wheel scroll down to ICDR diagnosis & care plan
 * 3. Progressive wheel scroll to Recent Patients ledger
 * 4. Progressive wheel scroll back to Preview PDF button
 * 5. Opens 1-Page PDF Report modal & scrolls INSIDE the report
 * 6. Closes the PDF modal completely
 * 7. Natural wheel scroll back to top of portal
 */
async function showcaseScreeningResults() {
  await sleep(900);

  // ── Step 1: Progressive wheel scroll down to Ingested Scan & green IQA badge ──
  await humanWheelScroll(240);
  await sleep(700);

  const retinaWrapper = document.querySelector('.image-canvas-wrapper');
  if (retinaWrapper) {
    await moveCursorTo(retinaWrapper, 0, -40);
    await sleep(1800);
  }

  // ── Step 2: Progressive wheel scroll down to Clinical Triage & Diagnosis card ─
  await humanWheelScroll(280);
  await sleep(700);

  const diagnosisCard = document.querySelector('.card-step[style*="border: 1.5px solid"]');
  if (diagnosisCard) {
    await moveCursorTo(diagnosisCard, 0, -10);
    await sleep(2200);
  }

  // ── Step 3: Progressive wheel scroll to Recent Patients ledger on right ──────
  await humanWheelScroll(220);
  await sleep(700);

  const recentPatients = document.querySelector('.widget-recent-card');
  if (recentPatients) {
    await moveCursorTo(recentPatients, 0, -20);
    await sleep(2000);
  }

  // ── Step 4: Accurately center the Preview PDF button and open the report modal ─
  const previewPdfBtn = document.getElementById('btn-preview-pdf');
  if (previewPdfBtn) {
    await sleep(400);
    await humanClick(previewPdfBtn);
    await sleep(1800);

    // Scroll INSIDE the modal body to showcase the full clinical certificate
    const modalBody = document.querySelector('.modal-body');
    if (modalBody) {
      modalBody.scrollBy({ top: 260, behavior: 'smooth' });
      await sleep(2200);
      modalBody.scrollBy({ top: 260, behavior: 'smooth' });
      await sleep(2400);
      modalBody.scrollTo({ top: 0, behavior: 'smooth' });
      await sleep(1400);
    } else {
      await sleep(3500);
    }

    // Close the PDF modal cleanly
    const closeBtn = document.getElementById('btn-close-pdf-modal') || document.querySelector('.modal-header button');
    if (closeBtn) {
      await humanClick(closeBtn);
    }

    // GUARANTEE: Ensure modal is 100% closed before doing anything else
    let closeWait = 0;
    while (document.querySelector('.modal-overlay') && closeWait < 6) {
      closeWait++;
      const fallbackClose = document.getElementById('btn-close-pdf-modal-footer') || document.getElementById('btn-close-pdf-modal');
      if (fallbackClose) fallbackClose.click();
      await sleep(300);
    }
    await sleep(600);
  }

  // ── Step 5: Natural wheel scroll back to top of portal ───────────────────────
  await humanWheelScrollToTop();
  await sleep(1000);
}

/**
 * Executes a full-screen, natural human-paced demonstration of DRISHYA.
 */
export async function runFullHumanDemo() {
  if (window.__DRISHYA_DEMO_RUNNING__) return;
  window.__DRISHYA_DEMO_RUNNING__ = true;

  try {
    // ── Setup & Reset ─────────────────────────────────────────────────────────
    await humanWheelScrollToTop();
    await sleep(900);

    // Ensure we start on Health Worker Portal
    const hwBtn = document.getElementById('btn-health-worker-mode');
    if (hwBtn && !hwBtn.classList.contains('active')) {
      await humanClick(hwBtn);
      await sleep(700);
    }

    // ── Image 1: Sunita Patel (Normal Retina / Grade 0) ───────────────────────
    const clearBtn = document.querySelector('.btn-portal-clear');
    if (clearBtn) {
      await humanClick(clearBtn);
      await sleep(500);
    }

    // Fill registration details with authentic human typing cadence
    await humanType('#input-patient-name', 'Sunita Patel', 185);
    await humanType('#input-patient-age', '48', 195);
    await humanSelect('#input-patient-gender', 'Female');
    await humanType('#input-patient-phone', '+91 94123 45678', 170);
    await humanType('#input-patient-abha', '14-8921-3401-9210', 165);
    await sleep(700);

    // Ingest Normal Scan (Grade 0)
    await humanClick('#btn-sample-normal');
    await waitForImageLoaded();
    await sleep(2000);

    // Click Run AI Screening Analysis
    await humanClick('#btn-process-scan');

    // Wait for the 6-stage clinical pipeline to complete
    await waitForScreeningToFinish();

    // SHOWCASE EVERYTHING ON THE PAGE FOR IMAGE 1
    await showcaseScreeningResults();

    // ── Image 2: Rajesh Verma (Moderate NPDR / Grade 2 Referral) ──────────────
    const clearScanBtn = document.querySelector('.btn-portal-clear');
    if (clearScanBtn) {
      await humanClick(clearScanBtn);
      await sleep(600);
    }

    // Type patient 2 details with human cadence
    await humanType('#input-patient-name', 'Rajesh Verma', 185);
    await humanType('#input-patient-age', '61', 200);
    await humanSelect('#input-patient-gender', 'Male');
    await humanType('#input-patient-phone', '+91 98234 56789', 170);
    await humanType('#input-patient-abha', '91-4521-8890-1234', 165);
    await sleep(700);

    // Ingest Moderate DR Scan (Grade 2)
    await humanClick('#btn-sample-moderate');
    await waitForImageLoaded();
    await sleep(2000);

    // Run AI analysis
    await humanClick('#btn-process-scan');
    await waitForScreeningToFinish();

    // SHOWCASE EVERYTHING ON THE PAGE FOR IMAGE 2
    await showcaseScreeningResults();

    // ── Deep Technical Inspection: Inspector Mode (Judge Inspector) ───────────
    const judgeModeBtn = document.getElementById('btn-judge-inspector-mode');
    if (judgeModeBtn) {
      await humanClick(judgeModeBtn);
      await sleep(1600);

      // Natural wheel scroll down to canvas and explainability tabs
      await humanWheelScroll(240);
      await sleep(800);

      // Tab 1: (a) Original Retinal Scan
      const tabRaw = document.getElementById('tab-raw');
      if (tabRaw) {
        await humanClick(tabRaw);
        await sleep(2200); // Showcase raw camera acquisition
      }

      // Tab 2: (b) CLAHE Preprocessed Scan
      const tabPreprocessed = document.getElementById('tab-preprocessed');
      if (tabPreprocessed) {
        await humanClick(tabPreprocessed);
        await sleep(2200); // Showcase Ben Graham contrast normalization
      }

      // Tab 3: (c) Microaneurysms & Exudates Segmentation
      const tabLesions = document.getElementById('tab-lesions');
      if (tabLesions) {
        await humanClick(tabLesions);
        await sleep(1800); // Inspect morphological lesion contours

        // Operates the sideways scroll wheel / slider on Lesions overlay
        const slider = document.getElementById('slider-heatmap-opacity');
        if (slider) {
          await humanDragSlider(slider, 85, 900); // Intensify lesion contours
          await sleep(1200);
          await humanDragSlider(slider, 45, 800); // Moderate blend
          await sleep(1000);
        }
      }

      // Tab 4: (d) Grad-CAM Visual AI Evidence
      const tabGradcam = document.getElementById('tab-gradcam');
      if (tabGradcam) {
        await humanClick(tabGradcam);
        await sleep(1800); // Inspect Grad-CAM neural attention peaks

        // Operates the sideways scroll wheel / slider on Grad-CAM heatmap
        const slider = document.getElementById('slider-heatmap-opacity');
        if (slider) {
          await humanDragSlider(slider, 95, 1000); // Intensify attention heatmap
          await sleep(1400);
          await humanDragSlider(slider, 20, 900);  // Fade heatmap to reveal blood vessels underneath
          await sleep(1200);
          await humanDragSlider(slider, 70, 800);  // Settle at optimal diagnostic blend
          await sleep(1200);
        }
      }

      // Wheel scroll down to showcase the Biomarkers & Quantitative Phenotyping table
      await humanWheelScroll(240);
      await sleep(1800);

      // Open and showcase the Diagnostic Report in Judge Inspector Mode
      const inspectorReportBtn = document.getElementById('btn-preview-pdf-inspector');
      if (inspectorReportBtn) {
        await humanClick(inspectorReportBtn);
        await sleep(1800);

        // Scroll inside the PDF report modal to inspect clinical metrics
        const modalBody = document.querySelector('.modal-body');
        if (modalBody) {
          modalBody.scrollBy({ top: 250, behavior: 'smooth' });
          await sleep(2000);
          modalBody.scrollBy({ top: 250, behavior: 'smooth' });
          await sleep(2200);
          modalBody.scrollTo({ top: 0, behavior: 'smooth' });
          await sleep(1200);
        }

        // Close the PDF modal cleanly
        const closeBtn = document.getElementById('btn-close-pdf-modal') || document.querySelector('.modal-header button');
        if (closeBtn) {
          await humanClick(closeBtn);
        }

        let closeWait = 0;
        while (document.querySelector('.modal-overlay') && closeWait < 6) {
          closeWait++;
          const fallbackClose = document.getElementById('btn-close-pdf-modal-footer') || document.getElementById('btn-close-pdf-modal');
          if (fallbackClose) fallbackClose.click();
          await sleep(300);
        }
        await sleep(600);
      }

      // Wheel scroll back up to top
      await humanWheelScrollToTop();
      await sleep(1000);
    }

    // ── Patient Records Archive: Supabase Cloud Ledger ────────────────────────
    const recordsModeBtn = document.getElementById('btn-patient-records-mode');
    if (recordsModeBtn) {
      await humanClick(recordsModeBtn);
      await sleep(1800);

      // Wheel scroll down to showcase the metric stat cards and archive table
      await humanWheelScroll(220);
      await sleep(1000);

      // Click "🚨 Referrals" filter pill
      const referralPill = Array.from(document.querySelectorAll('button')).find((b) =>
        b.textContent.includes('Referrals')
      );
      if (referralPill) {
        await humanClick(referralPill);
        await sleep(2200);
      }

      // Search for Rajesh Verma
      const searchInput = document.querySelector("input[placeholder*='Search']");
      if (searchInput) {
        await humanType(searchInput, 'Rajesh', 190);
        await sleep(2400);
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
        await sleep(800);
      }

      // Return to All Records
      const allPill = Array.from(document.querySelectorAll('button')).find((b) =>
        b.textContent.includes('All Records')
      );
      if (allPill) {
        await humanClick(allPill);
        await sleep(1400);
      }

      // Wheel scroll down to showcase the complete table
      await humanWheelScroll(280);
      await sleep(2000);

      await humanWheelScrollToTop();
      await sleep(1000);
    }

    // ── Conclude & Return to Health Worker Portal ─────────────────────────────
    if (hwBtn) {
      await humanClick(hwBtn);
      await humanWheelScrollToTop();
      await sleep(1600);
    }
  } finally {
    removeCursor();
    window.__DRISHYA_DEMO_RUNNING__ = false;
  }
}
