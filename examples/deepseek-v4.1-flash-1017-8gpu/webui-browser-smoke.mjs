#!/usr/bin/env node

/**
 * Browser gate for this example's Open WebUI (run by `verify.sh chat-ui`):
 * two typed turns of one chat answer 323 and 324, an attached red PNG answers
 * Red, and a Japanese question with the gate's knowledge base selected by `#`
 * is answered from its scanned PDF with that document cited.
 */

import { chromium } from 'playwright';

const baseUrl = new URL(process.env.WEBUI_SMOKE_BASE_URL ?? 'http://127.0.0.1:3018');
const model = process.env.WEBUI_SMOKE_MODEL ?? 'deepseek-v4.1-flash-1017';
const knowledge = process.env.WEBUI_SMOKE_KNOWLEDGE;
const knowledgeQuery = process.env.WEBUI_SMOKE_KNOWLEDGE_QUERY;
const knowledgeSource = process.env.WEBUI_SMOKE_KNOWLEDGE_SOURCE;
const actionTimeoutMs = 20_000;
const navigationTimeoutMs = 60_000;
const responseTimeoutMs = 1_800_000;

let browser;
let page;
let currentStep = 'startup';

function invariant(condition, message) {
	if (!condition) throw new Error(message);
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
	const context = await browser.newContext({ serviceWorkers: 'block', locale: 'ja-JP' });
	page = await context.newPage();
	await step('open the no-auth chat', newChat);
	await step('the public model is offered', async () => {
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

try {
	await main();
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
