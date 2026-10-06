#!/usr/bin/env node

/**
 * Browser gate for this example's answer page (VCO-D18): send one request in
 * the page and require the answer on the always-verified model. The guarantee
 * panel is not checked until the guarantee is rebuilt.
 */

import { chromium } from 'playwright';

const baseUrl = process.env.PLAYGROUND_SMOKE_BASE_URL ?? 'http://127.0.0.1:3013';
const responseTimeoutMs = Number(process.env.PLAYGROUND_SMOKE_RESPONSE_TIMEOUT_MS ?? 1_800_000);

const browser = await chromium.launch();
try {
	const page = await browser.newPage();
	const failures = [];
	const models = [];
	page.on('requestfailed', (request) => failures.push(request.url()));
	page.on('request', (request) => {
		if (request.url().endsWith('/v1/chat/completions')) models.push(request.postDataJSON()?.model);
	});
	await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 30_000 });
	await page.fill('#prompt', 'List three primary colors as a comma-separated line, nothing else.');
	await page.click('#send');
	const card = page.locator('.answer, .card.err').first();
	await card.waitFor({ timeout: responseTimeoutMs });
	const answer = (await page.locator('.answer').first().textContent().catch(() => '')).trim();
	if (!answer) throw new Error(`the page shows no answer: ${(await card.textContent()).trim()}`);
	if (failures.length) throw new Error(`same-origin requests failed: ${failures.join(', ')}`);
	// The answer page always takes the verified path.
	if (JSON.stringify(models) !== JSON.stringify(['kairyu-verified-always'])) {
		throw new Error(`answer page used models ${JSON.stringify(models)}`);
	}
	console.log(JSON.stringify({ ok: true, answer: answer.slice(0, 120) }));
} finally {
	await browser.close();
}
