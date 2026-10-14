"""Web pages as text for the web environments: DOM snapshot, page rendering, actions at a point, typing candidates.

The snapshot runs in the page (SNAPSHOT_JS) and returns the visible elements in document order, each with
its box: the part of the element that shows inside the area actions can reach, in page pixels. The rendering
lists them as an indented outline (box, tag, #id, .classes, visible text in quotes, then state). An action
names a point: CLICK_JS acts on the element on top there and SCROLL_JS scrolls the scroll box under it.
Nothing here is task-specific.

Visible: not display:none, not visibility:hidden, not fully transparent and larger than 1x1 px (so
screen-reader-only helpers stay out). The options of a <select> are listed with their list even when
the list is closed (a closed list's options have no box). Elements scrolled out of a scroll box, clipped
away or outside the area are listed too, marked out of view with the box they would take, so the whole
page can be read, but only what is in view can be clicked. Before an episode starts the harness loads the
images the style sheets use (icons drawn with CSS have no size until then), so what is visible does not
depend on loading speed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ...schema import MAX_OPTIONS

# Shared by SNAPSHOT_JS and SCROLL_JS: an element whose content scrolls vertically and does not fit. Text
# areas are left out: their whole text is listed, so scrolling them is never needed.
SCROLL_BOX_JS = r"""
function scrollBox(el) {
  return el.tagName !== 'TEXTAREA' && el.scrollHeight > el.clientHeight + 1 && /^(auto|scroll)$/.test(getComputedStyle(el).overflowY);
}
"""

# Called with the width and height of the area actions can reach (from the page's top-left corner). Returns
# {"nodes": [...], "busy": bool} and keeps the listed elements in window.__jvNodes, so an action can name the
# element it reached by its index in the latest snapshot.
SNAPSHOT_JS = SCROLL_BOX_JS + r"""
var W = arguments[0], H = arguments[1];
var SKIP = {'reward-display': 1, 'sync-task-cover': 1, 'click-canvas': 1, 'query': 1, 'attention-canvas': 1};
var IGNORE = {script: 1, style: 1, noscript: 1, br: 1, link: 1, meta: 1, template: 1};
var TEXTUAL = /^(text|search|password|email|url|tel|number|date|time|datetime-local|month|week)$/;
var nodes = [], els = [];
function norm(s) { return (s || '').replace(/\s+/g, ' ').trim(); }
function ownText(el) {
  for (var c = el.firstChild; c; c = c.nextSibling) if (c.nodeType === 3 && /\S/.test(c.data)) return true;
  return false;
}
function rounded(x0, y0, x1, y1) { var x = Math.round(x0), y = Math.round(y0); return [x, y, Math.round(x1) - x, Math.round(y1) - y]; }
// The part of el inside the area and inside every ancestor that clips it (overflow other than visible), or null.
function shownBox(el) {
  var r = el.getBoundingClientRect(), x0 = Math.max(r.left, 0), y0 = Math.max(r.top, 0), x1 = Math.min(r.right, W), y1 = Math.min(r.bottom, H);
  var position = getComputedStyle(el).position;
  for (var p = el.parentElement; p && p !== document.body && position !== 'fixed'; p = p.parentElement) {
    var cs = getComputedStyle(p);
    if (position === 'absolute' && cs.position === 'static') continue;  // not in its chain of containing blocks: no clipping
    if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') {
      var q = p.getBoundingClientRect(), left = q.left + p.clientLeft, top = q.top + p.clientTop;
      x0 = Math.max(x0, left); y0 = Math.max(y0, top); x1 = Math.min(x1, left + p.clientWidth); y1 = Math.min(y1, top + p.clientHeight);
    }
    position = cs.position;
  }
  return x1 - x0 >= 1 && y1 - y0 >= 1 ? rounded(x0, y0, x1, y1) : null;
}
function add(el, depth, noBox) {
  var tag = el.tagName.toLowerCase();
  var node = {tag: tag, depth: depth};
  if (el.id) node.id = el.id;
  var cls = typeof el.className === 'string' ? el.className : (el.className && el.className.baseVal) || '';
  if (norm(cls)) node.cls = norm(cls);
  if (tag === 'input') {
    node.type = (el.type || 'text').toLowerCase();
    if (node.type === 'checkbox' || node.type === 'radio') node.checked = el.checked;
    else if (TEXTUAL.test(node.type)) node.value = el.value;
    else if (el.value) node.label = el.value;  // buttons: the value is the visible label
  } else if (tag === 'textarea') {
    node.value = el.value;
  } else if (tag === 'select') {
    node.multiple = el.multiple;
  } else if (tag === 'option') {
    node.selected = el.selected;
    node.text = norm(el.text);
  }
  if (tag !== 'textarea' && tag !== 'select' && tag !== 'option' && (el.children.length === 0 || ownText(el))) {
    var t = norm(el.innerText !== undefined ? el.innerText : el.textContent);
    if (t) node.text = t;
  }
  ['placeholder', 'title', 'aria-label', 'alt'].forEach(function (a) { var v = el.getAttribute(a); if (v && norm(v)) node[a] = norm(v); });
  if (el.disabled) node.disabled = true;
  if (el.readOnly && (tag === 'input' || tag === 'textarea')) node.readonly = true;
  if (el.isContentEditable && !(el.parentElement && el.parentElement.isContentEditable)) node.editable = true;
  if (document.activeElement === el && el !== document.body) node.focused = true;
  if (!noBox) {
    var part = shownBox(el), r = el.getBoundingClientRect();
    node.box = part || rounded(r.left, r.top, r.right, r.bottom);
    if (!part) node.offscreen = true;
  }
  if (scrollBox(el)) {
    if (el.scrollTop > 0) node.scroll_up = true;
    if (el.scrollTop + el.clientHeight < el.scrollHeight - 1) node.scroll_down = true;
  }
  nodes.push(node);
  els.push(el);
  return node;
}
function walk(el, depth) {
  if (SKIP[el.id] || IGNORE[el.tagName.toLowerCase()]) return;
  var cs = window.getComputedStyle(el);
  if (cs.display === 'none' || parseFloat(cs.opacity) === 0) return;
  var r = el.getBoundingClientRect();
  var shown = cs.visibility !== 'hidden' && cs.visibility !== 'collapse' && r.width > 1 && r.height > 1;
  var node = shown ? add(el, depth) : null;
  var before = nodes.length;
  if (shown && el.tagName === 'SELECT') {
    var closed = !el.multiple && el.size <= 1;  // a drop-down list: its options have no place on the page
    for (var i = 0; i < el.options.length; i++) {
      var o = el.options[i];
      if (!o.hidden && window.getComputedStyle(o).display !== 'none') add(o, depth + 1, closed);
    }
  } else {
    for (var k = 0; k < el.children.length; k++) walk(el.children[k], shown ? depth + 1 : depth);
  }
  if (node) node.leaf = nodes.length === before;
}
for (var k = 0; k < document.body.children.length; k++) walk(document.body.children[k], 0);
window.__jvNodes = els;
var busy = !!(window.jQuery && window.jQuery.timers && window.jQuery.timers.length) || !!(window.__jvPending && window.__jvPending())
  || Array.prototype.some.call(document.images, function (im) { return !im.complete; });
return {nodes: nodes, busy: busy};
"""

# Installed once per page before the episode starts: counts pending short timeouts (widget delays of up
# to 2 s, e.g. an autocomplete list or a date picker placed by a deferred callback), so the harness can
# wait for them before it reads the page.
TIMERS_JS = r"""
if (!window.__jvPending) {
  var pending = {}, set = window.setTimeout, clear = window.clearTimeout;
  window.__jvPending = function () { return Object.keys(pending).length; };
  window.setTimeout = function (fn, delay) {
    var args = Array.prototype.slice.call(arguments, 2), id;
    id = set(function () { delete pending[id]; if (typeof fn === 'function') fn.apply(window, args); else (0, eval)(fn); }, delay);
    if (!(delay > 2000)) pending[id] = true;
    return id;
  };
  window.clearTimeout = function (id) { delete pending[id]; return clear(id); };
}
"""

# Started once per page before the episode starts: loads every image the page's style sheets refer to
# (icons drawn with CSS content:url(...) have no size until their image has loaded), so new elements get
# their final size at once. IMAGES_READY_JS reports when all have loaded.
PRELOAD_JS = r"""
var urls = [];
for (var i = 0; i < document.styleSheets.length; i++) {
  var sheet = document.styleSheets[i], rules;
  try { rules = sheet.cssRules; } catch (e) { continue; }
  if (!rules) continue;
  for (var j = 0; j < rules.length; j++) {
    var re = /url\(\s*["']?([^"')]+)["']?\s*\)/g, m;
    while ((m = re.exec(rules[j].cssText))) {
      try { var u = new URL(m[1], sheet.href || document.baseURI).href; if (!/^data:/.test(u) && urls.indexOf(u) < 0) urls.push(u); } catch (e) {}
    }
  }
}
window.__jvImages = urls.map(function (u) { var im = new Image(); im.src = u; return im; });
return urls.length;
"""
IMAGES_READY_JS = "return (window.__jvImages || []).every(function (im) { return im.complete; });"

# Click the element on top at point (x, y), as MiniWoB's element click does (click event, then focus). An
# <option> is chosen in its list instead; in a list that allows several selections it is toggled. Returns the
# index in the latest snapshot of the element reached (or of its nearest listed ancestor), "instruction" for
# MiniWoB's task instruction, or null for the page background.
CLICK_JS = r"""
var el = document.elementFromPoint(arguments[0], arguments[1]);
if (!el) return null;
var named = el;
while (named && window.__jvNodes.indexOf(named) < 0) named = named.parentElement;
var hit = el.closest('#query') ? 'instruction' : named ? window.__jvNodes.indexOf(named) : null;
if (el.tagName === 'OPTION') {
  var list = el.parentElement && el.parentElement.tagName === 'SELECT' ? el.parentElement : el.closest('select');
  if (list && list.multiple) el.selected = !el.selected; else el.selected = true;
  if (list) {
    list.focus();
    list.dispatchEvent(new Event('input', {bubbles: true}));
    list.dispatchEvent(new Event('change', {bubbles: true}));
  }
} else if (el instanceof SVGElement) {
  ['mousedown', 'mouseup', 'click'].forEach(function (t) { el.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window})); });
} else {
  el.click();
  if (typeof el.focus === 'function') el.focus();
}
return hit;
"""

# Scroll the innermost scroll box under point (x, y) by 7/8 of its visible height, as Page Down and Page Up do:
# down for direction 1, up for -1. Returns the scroll box's index in the latest snapshot, or null if there is none.
SCROLL_JS = SCROLL_BOX_JS + r"""
for (var el = document.elementFromPoint(arguments[0], arguments[1]); el && el !== document.body; el = el.parentElement) {
  if (scrollBox(el)) {
    el.scrollTop += arguments[2] * Math.max(1, Math.floor(el.clientHeight * 7 / 8));
    var i = window.__jvNodes.indexOf(el);
    return i < 0 ? null : i;
  }
}
return null;
"""

# Before typing: select the field's current text so the typed text replaces it.
SELECT_TEXT_JS = r"""
var el = document.activeElement;
if (!el) return false;
try { if (typeof el.select === 'function') { el.select(); return true; } } catch (e) {}
if (el.isContentEditable) { document.execCommand('selectAll', false, null); return true; }
return false;
"""

# jQuery UI's layout and theming classes; they carry no meaning for the task and are left out.
DECORATIVE = re.compile(r"^ui-(corner-.+|widget(-content|-header)?|helper-.+|front|draggable(-handle)?|resizable(-.+)?|state-default|priority-.+|button-icon-space)$")
CONTROL_TAGS = {"input", "textarea", "select", "option", "button", "a"}
PAGE_TEXT_MAX = 150  # page texts longer than this are not offered as typing candidates
GENERATED_ID = re.compile(r"^ui-id-\d+$")  # jQuery UI numbers widgets as it creates them (the count depends on timing)


def classes(node: dict[str, Any]) -> list[str]:
    return [c for c in node.get("cls", "").split() if not DECORATIVE.match(c)]


def element_id(node: dict[str, Any]) -> str:
    return "" if GENERATED_ID.match(node.get("id", "")) else node.get("id", "")


def listed(node: dict[str, Any]) -> bool:
    """Elements the outline shows: containers (the page's structure), leaves with visible text, form controls, links
    and buttons, and leaves something identifies (icons)."""
    if not node.get("leaf") or node.get("text") or node["tag"] in CONTROL_TAGS or node.get("editable"):
        return True
    return bool(element_id(node) or classes(node) or node.get("title") or node.get("aria-label") or node.get("alt"))


def describe(node: dict[str, Any]) -> str:
    """One element as a line: tag[type]#id.classes "text" and its state."""
    tag = node["tag"] + (f"[{node['type']}]" if node.get("type") else "")
    out = tag + (f"#{element_id(node)}" if element_id(node) else "") + "".join(f".{c}" for c in classes(node))
    text = node.get("text") or node.get("label")
    if text:
        out += " " + json.dumps(text, ensure_ascii=False)
    if "value" in node:
        out += " value=" + json.dumps(node["value"], ensure_ascii=False)
    for attr in ("placeholder", "title", "aria-label", "alt"):
        if node.get(attr) and node.get(attr) != node.get("text"):
            out += f" {attr}=" + json.dumps(node[attr], ensure_ascii=False)
    state = []
    if "checked" in node:
        state.append("checked" if node["checked"] else "unchecked")
    if node.get("selected"):
        state.append("selected")
    if node.get("multiple"):
        state.append("several options can be selected")
    if node.get("disabled"):
        state.append("disabled")
    if node.get("readonly"):
        state.append("read-only")
    if node.get("editable"):
        state.append("editable text")
    if node.get("focused"):
        state.append("focused")
    if node.get("offscreen"):
        state.append("out of view")
    if node.get("scroll_up"):
        state.append("can scroll up")
    if node.get("scroll_down"):
        state.append("can scroll down")
    return out + (f" ({', '.join(state)})" if state else "")


def short(node: dict[str, Any], max_text: int = 40) -> str:
    """An element as named in the list of past actions: tag, id, classes and text, without its state."""
    out = node["tag"] + (f"[{node['type']}]" if node.get("type") else "") + (f"#{element_id(node)}" if element_id(node) else "")
    out += "".join(f".{c}" for c in classes(node))
    text = node.get("text") or node.get("label") or node.get("placeholder") or node.get("title") or node.get("aria-label")
    if text:
        out += " " + json.dumps(text if len(text) <= max_text else text[: max_text - 1] + "…", ensure_ascii=False)
    return out


def render(nodes: list[dict[str, Any]]) -> str:
    """The page as an indented outline of the listed elements, each led by its box (x, y, width, height)."""
    lines = []
    for node in nodes:
        if listed(node):
            box = f"({', '.join(str(v) for v in node['box'])}) " if "box" in node else ""
            lines.append("  " * node["depth"] + box + describe(node))
    return "\n".join(lines)


QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”|(?<!\w)\'([^\']+)\'(?!\w)')
TEXT_TYPES = {"text", "search", "password", "email", "url", "tel", "number", "date", "time", "datetime-local", "month", "week"}


def accepts_text(node: dict[str, Any]) -> bool:
    """An element the keyboard can fill: a text-like input, a text area, editable text or a list (typing an option's
    text selects that option), not read-only or disabled."""
    if node.get("readonly") or node.get("disabled"):
        return False
    return (node["tag"] == "input" and node.get("type") in TEXT_TYPES) or node["tag"] in ("textarea", "select") or bool(node.get("editable"))


def typing_candidates(utterance: str, field_values: list[str], nodes: list[dict[str, Any]]) -> list[str]:
    """Texts the model may type, in a fixed order: quoted phrases of the instruction, MiniWoB's field
    values for the instruction, then the texts and field values shown on the page (up to PAGE_TEXT_MAX
    characters). Field values are kept exactly as they are, including trailing spaces."""
    out: list[str] = []

    def add(text: str | None) -> None:
        if text and text not in out:
            out.append(text)

    for match in QUOTED.finditer(utterance):
        add(next(g for g in match.groups() if g))
    for value in field_values:
        add(value)
    for node in nodes:
        for text in (node.get("text"), node.get("value")):
            if text and len(text) <= PAGE_TEXT_MAX:
                add(text)
    return out[: MAX_OPTIONS - 1]  # one option is left for clearing the field
