#!/usr/bin/env node

/**
 * Browser gate for this example's Open WebUI (run by `verify.sh chat-ui`):
 * two typed turns of one chat answer 323 and 324, an attached red PNG answers
 * Red, and a Japanese question with the gate's knowledge base selected by `#`
 * is answered from its scanned PDF with that document cited.
 *
 * With WEBUI_SMOKE_MODE=office (run by `verify.sh office`), each Word, Excel
 * and PowerPoint fixture is attached with the clip, asked about, then edited
 * by the code interpreter; a result the reply delivers is saved into /results
 * for the gate to check. A reply delivers a file when the download tool
 * attached it (the file item above the text) or the text links it; it is
 * saved the way a user does: the text link if there is one, else the file
 * item and the link in its dialog. At its official sampling the model
 * sometimes answers without calling the download tool: such a format is
 * recorded as a model miss, and at least one format must be delivered.
 *
 * With WEBUI_SMOKE_MODE=download (run by `verify.sh file-download`), the
 * owner's request for an Excel template is asked on a desktop screen and on
 * an emulated phone, each in up to three new chats until a reply delivers
 * the file (misses recorded); the file is saved from that reply, then again
 * from the same reply after a reload.
 *
 * The Chat UI requires sign-in: every browser context starts with the admin
 * session token the gate obtained (WEBUI_SMOKE_TOKEN), in local storage for
 * the page and as the cookie the download links need.
 */

import { chromium, devices } from 'playwright';

const baseUrl = new URL(process.env.WEBUI_SMOKE_BASE_URL ?? 'http://127.0.0.1:3020');
const model = process.env.WEBUI_SMOKE_MODEL ?? 'gemma-4-31b';
const token = process.env.WEBUI_SMOKE_TOKEN;
const knowledge = process.env.WEBUI_SMOKE_KNOWLEDGE;
const knowledgeQuery = process.env.WEBUI_SMOKE_KNOWLEDGE_QUERY;
const knowledgeSource = process.env.WEBUI_SMOKE_KNOWLEDGE_SOURCE;
const mode = process.env.WEBUI_SMOKE_MODE ?? 'chat';
const officeCases = JSON.parse(process.env.WEBUI_SMOKE_OFFICE ?? '{}');
const downloadRequest = process.env.WEBUI_SMOKE_DOWNLOAD_REQUEST;
const actionTimeoutMs = 20_000;
const navigationTimeoutMs = 60_000;
const responseTimeoutMs = 1_800_000;

let browser;
let page;
let currentStep = 'startup';

function invariant(condition, message) {
	if (!condition) throw new Error(message);
}

// A browser context already signed in as the gate's user.
async function signedInContext(options) {
	invariant(token, 'WEBUI_SMOKE_TOKEN is not set');
	const context = await browser.newContext(options);
	await context.addInitScript((value) => localStorage.setItem('token', value), token);
	await context.addCookies([{ name: 'token', value: token, url: baseUrl.origin }]);
	return context;
}

async function step(name, operation) {
	currentStep = name;
	const started = Date.now();
	const result = await operation();
	console.log(JSON.stringify({ step: name, seconds: (Date.now() - started) / 1000 }));
	return result;
}

// Open WebUI's corner notices and its first-run "what's new" dialog can cover
// the controls; they are informational, so the gate closes them first.
async function dismissNotices() {
	const dialogs = page.locator('div[role="dialog"][aria-modal="true"]');
	for (let attempt = 0; attempt < 3 && (await dialogs.count()) > 0; attempt += 1) {
		await page.keyboard.press('Escape');
		await dialogs.first().waitFor({ state: 'detached', timeout: 2_000 }).catch(() => {});
		if ((await dialogs.count()) > 0) {
			await dialogs.first().locator('button').last().click({ timeout: actionTimeoutMs }).catch(() => {});
		}
	}
	await page
		.locator('div.absolute.bottom-8.right-8.z-50')
		.evaluateAll((elements) => elements.forEach((element) => element.remove()));
}

async function newChat() {
	await page.goto(baseUrl.href, { waitUntil: 'domcontentloaded', timeout: navigationTimeoutMs });
	await page.locator('#chat-input').waitFor({ state: 'visible', timeout: navigationTimeoutMs });
	await dismissNotices();
}

// Send what is in the input and return the new assistant message's text.
async function sendAndRead() {
	const items = page.locator('ul[role="log"] [role="listitem"]');
	const before = await items.count();
	await page.locator('#send-message-button').click({ timeout: actionTimeoutMs });
	await page.waitForFunction(
		(count) => document.querySelectorAll('ul[role="log"] [role="listitem"]').length >= count,
		before + 2,
		{ timeout: responseTimeoutMs }
	);
	const item = items.nth(before + 1);
	await item.locator('.copy-response-button').waitFor({ state: 'visible', timeout: responseTimeoutMs });
	const text = (await item.innerText()).trim();
	invariant(text.length > 0, 'empty answer');
	return { item, text };
}

async function ask(prompt) {
	await page.locator('#chat-input').fill(prompt);
	return sendAndRead();
}

async function main() {
	browser = await chromium.launch({ headless: true });
	const context = await signedInContext({ serviceWorkers: 'block', locale: 'ja-JP' });
	page = await context.newPage();
	await step('open the signed-in chat', newChat);
	await step('only the chat model is offered (no embedding model)', async () => {
		const ids = await page.evaluate(async () => {
			const response = await fetch('/api/models', {
				headers: { Authorization: `Bearer ${localStorage.token}` }
			});
			return (await response.json()).data.map((row) => row.id).sort();
		});
		invariant(JSON.stringify(ids) === JSON.stringify([model]), `models offered: ${JSON.stringify(ids)}`);
	});
	await step('a typed turn answers 323', async () => {
		const { text } = await ask('What is 17 * 19? Reply with the number only.');
		invariant(/\b323\b/.test(text), `first turn: ${text}`);
	});
	await step('the next turn of the same chat answers 324', async () => {
		const { text } = await ask('Add 1 to that number. Reply with the number only.');
		invariant(/\b324\b/.test(text), `second turn: ${text}`);
	});
	await step('an attached red PNG answers Red', async () => {
		await newChat();
		const upload = page.waitForResponse(
			(response) => response.url().includes('/api/v1/files') && response.request().method() === 'POST',
			{ timeout: navigationTimeoutMs }
		);
		await page.locator('input[type="file"]').first().setInputFiles('/fixtures/red.png');
		const response = await upload;
		invariant(response.ok(), `image upload returned HTTP ${response.status()}`);
		const { text } = await ask('What color is this image? Answer with one English word.');
		invariant(/\bred\b/i.test(text), `image turn: ${text}`);
	});
	if (knowledge) {
		await step('a knowledge base selected by # answers from its scanned PDF', async () => {
			await newChat();
			const input = page.locator('#chat-input');
			await input.click();
			await page.keyboard.type(`#${knowledge}`, { delay: 30 });
			const option = page.getByRole('button', { name: new RegExp(knowledge) }).first();
			await option.waitFor({ state: 'visible', timeout: actionTimeoutMs });
			await option.click({ timeout: actionTimeoutMs });
			await page.getByText(knowledge).first().waitFor({ state: 'visible', timeout: actionTimeoutMs });
			await input.click();
			await page.keyboard.press('Control+A');
			await page.keyboard.type(knowledgeQuery);
			const { item, text } = await sendAndRead();
			invariant(/(四十二|42)/.test(text), `knowledge answer has no storage temperature: ${text}`);
			const cited = text.includes(knowledgeSource) || (await item.getByText(knowledgeSource).count()) > 0;
			invariant(cited, `knowledge answer does not cite ${knowledgeSource}: ${text}`);
			console.log(JSON.stringify({ knowledge_answer: text.slice(0, 600) }));
		});
	}
	console.log('WEBUI BROWSER SMOKE PASS');
}

async function apiJson(path) {
	return page.evaluate(async (target) => {
		const response = await fetch(target, {
			headers: { Authorization: `Bearer ${localStorage.token}` }
		});
		return response.json();
	}, path);
}

// Attach a file with the clip and wait until Open WebUI has extracted it.
async function attach(path) {
	const upload = page.waitForResponse(
		(response) =>
			/\/api\/v1\/files\/?(\?|$)/.test(new URL(response.url()).pathname + new URL(response.url()).search) &&
			response.request().method() === 'POST',
		{ timeout: navigationTimeoutMs }
	);
	await page.locator('input[type="file"]').first().setInputFiles(path);
	const response = await upload;
	invariant(response.ok(), `upload of ${path} returned HTTP ${response.status()}`);
	const { id } = await response.json();
	// Images are not extracted; only documents wait for their processing.
	if (/\.(png|jpe?g|webp|gif)$/i.test(path)) return id;
	const deadline = Date.now() + 300_000;
	for (;;) {
		const { status } = await apiJson(`/api/v1/files/${id}/process/status`);
		if (status === 'completed') return id;
		invariant(status !== 'failed' && Date.now() < deadline, `processing ${path}: ${status}`);
		await page.waitForTimeout(1000);
	}
}

// The file item the download tool attached to a reply (shown above its
// text); clicking it opens a dialog whose title links the stored file.
function attachedFile(item) {
	return item.locator('button.w-60').filter({ hasText: /\.(docx|xlsx|pptx)\b/i }).first();
}

// Whether the reply delivers a file: a link in its text or an attached file.
async function deliversFile(item) {
	return (
		(await item.locator('a[href*="/api/v1/files/"]').count()) > 0 || (await attachedFile(item).count()) > 0
	);
}

// Save the file the reply delivers (the file-download tool's
// /api/v1/files/<id>/content link) as /results/<saveAs>, the way a user
// clicks or taps it: the reply's link if its text has one, otherwise the
// attached file item and the link in its dialog. The link opens a new tab
// that downloads the file.
async function downloadFromReply(item, saveAs, { tap = false } = {}) {
	const text = (await item.innerText()).trim();
	invariant(!/sandbox:/i.test(text), `the reply still writes a sandbox: link: ${text.slice(0, 400)}`);
	let link = item.locator('a[href*="/api/v1/files/"]').last();
	if ((await link.count()) === 0) {
		const file = attachedFile(item);
		invariant((await file.count()) > 0, `the reply delivers no file: ${text.slice(0, 600)}`);
		await (tap ? file.tap({ timeout: actionTimeoutMs }) : file.click({ timeout: actionTimeoutMs }));
		link = page.locator('div[role="dialog"] a[href*="/api/v1/files/"]').first();
		await link.waitFor({ timeout: actionTimeoutMs });
	}
	const href = await link.getAttribute('href');
	// The link opens a new tab whose response is an attachment; Chromium hands
	// the download to the opening page and closes the tab, another browser may
	// keep it on the tab, so take whichever reports it.
	const onPage = page.waitForEvent('download', { timeout: navigationTimeoutMs });
	const onTab = page
		.waitForEvent('popup', { timeout: navigationTimeoutMs })
		.then((tab) => tab.waitForEvent('download', { timeout: navigationTimeoutMs }));
	await (tap ? link.tap({ timeout: actionTimeoutMs }) : link.click({ timeout: actionTimeoutMs }));
	const download = await Promise.any([onPage, onTab]);
	onPage.catch(() => {});
	onTab.catch(() => {});
	await download.saveAs(`/results/${saveAs}`);
	console.log(JSON.stringify({ saved: saveAs, href, suggested: download.suggestedFilename() }));
	await page.keyboard.press('Escape');
	return href;
}

// Ask for the Excel file in a new chat until a reply delivers it (at most
// three chats). A reply that delivers no file is the model not calling the
// download tool at its official sampling; it is recorded, not judged.
async function askForFile(label) {
	for (let attempt = 1; attempt <= 3; attempt += 1) {
		await newChat();
		const { item, text } = await ask(downloadRequest);
		console.log(JSON.stringify({ [`${label}_answer`]: text.slice(0, 500), attempt }));
		if (await deliversFile(item)) return item;
		console.log(JSON.stringify({ model_miss: label, attempt }));
	}
	throw new Error(`${label}: three replies delivered no file`);
}

// The last assistant message of the open chat.
function lastReply() {
	return page.locator('ul[role="log"] [role="listitem"]').last();
}

async function officeMain() {
	browser = await chromium.launch({ headless: true });
	const context = await signedInContext({
		serviceWorkers: 'block',
		locale: 'en-US',
		acceptDownloads: true,
		viewport: { width: 1600, height: 1000 }
	});
	page = await context.newPage();
	const misses = [];
	for (const [format, item] of Object.entries(officeCases)) {
		await step(`${format}: attach ${item.fixture}`, async () => {
			await newChat();
			invariant(
				(await page.locator('button[aria-label="Disable Code Interpreter"]').count()) > 0,
				'the code interpreter is not on by default'
			);
			await attach(`/fixtures/${item.fixture}`);
		});
		await step(`${format}: the attachment is read`, async () => {
			await page.locator('#chat-input').fill(item.question);
			const { text } = await sendAndRead();
			invariant(new RegExp(item.answer).test(text), `${format} answer: ${text.slice(0, 600)}`);
			console.log(JSON.stringify({ format, read_answer: text.slice(0, 300) }));
		});
		await step(`${format}: the code interpreter edits it into ${item.result}`, async () => {
			await page.locator('#chat-input').fill(item.edit);
			const { item: reply, text } = await sendAndRead();
			console.log(JSON.stringify({ format, edit_answer: text.slice(0, 400) }));
			invariant(/execute_code/.test(text), `${format}: the code interpreter did not run: ${text.slice(0, 400)}`);
			if (!(await deliversFile(reply))) {
				misses.push(format);
				console.log(JSON.stringify({ model_miss: format }));
				return;
			}
			await downloadFromReply(reply, item.result);
		});
	}
	invariant(misses.length < Object.keys(officeCases).length, 'no format delivered its file');
	// The last chat used the code interpreter; an image there now travels
	// beside that tool history (Kairyu used to refuse the turn). Any colour
	// word shows the image reached the model; which one is the model's
	// accuracy, which beside tool history varies on vLLM directly too.
	await step('an image in a chat that used the code interpreter is answered', async () => {
		await attach('/fixtures/red.png');
		const { text } = await ask('What color is this image? Answer with one English word.');
		invariant(!/cannot be combined|invalid_request_error/i.test(text), `image turn refused: ${text}`);
		invariant(
			/\b(red|crimson|scarlet|maroon|white|black|blue|green|yellow|orange|pink|purple|gray|grey|brown)\b/i.test(text),
			`image turn names no colour: ${text.slice(0, 300)}`
		);
		console.log(JSON.stringify({ image_after_tools: text.slice(0, 200) }));
	});
	console.log('WEBUI OFFICE SMOKE PASS');
}

async function downloadMain() {
	browser = await chromium.launch({ headless: true });
	const desktop = await signedInContext({
		serviceWorkers: 'block',
		locale: 'ja-JP',
		acceptDownloads: true,
		viewport: { width: 1600, height: 1000 }
	});
	page = await desktop.newPage();
	let href;
	await step('desktop: the owner\'s request gets a download link that saves the file', async () => {
		href = await downloadFromReply(await askForFile('desktop'), 'desktop.xlsx');
	});
	await step('desktop: after a reload the same link saves the same file', async () => {
		await page.reload({ waitUntil: 'domcontentloaded', timeout: navigationTimeoutMs });
		await page.locator('ul[role="log"] [role="listitem"]').first().waitFor({ timeout: navigationTimeoutMs });
		// The reloaded page opens Open WebUI's "what's new" dialog again.
		await dismissNotices();
		const again = await downloadFromReply(lastReply(), 'desktop-after-reload.xlsx');
		invariant(again === href, `the link changed after the reload: ${href} -> ${again}`);
	});
	await desktop.close();
	const phone = await signedInContext({
		...devices['iPhone 13'],
		serviceWorkers: 'block',
		locale: 'ja-JP',
		acceptDownloads: true
	});
	page = await phone.newPage();
	await step('phone: tapping the reply\'s link saves the file', async () => {
		await downloadFromReply(await askForFile('phone'), 'phone.xlsx', { tap: true });
	});
	console.log('WEBUI DOWNLOAD SMOKE PASS');
}

try {
	await (mode === 'office' ? officeMain() : mode === 'download' ? downloadMain() : main());
} catch (error) {
	console.error(`WEBUI BROWSER SMOKE FAIL [step=${currentStep}]\n${error?.stack ?? error}`);
	if (page) {
		await page.screenshot({ path: '/tmp/failure.png', fullPage: true }).catch(() => {});
		console.error((await page.locator('body').innerText().catch(() => '')).slice(-3000));
	}
	process.exitCode = 1;
} finally {
	await browser?.close().catch(() => {});
}
