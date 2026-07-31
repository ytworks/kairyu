#!/usr/bin/env node

/** Browser proof that an actual Open WebUI file input reaches Kairyu image chat. */

import { chromium } from 'playwright';

const baseUrl = new URL(process.env.WEBUI_BASE_URL ?? 'http://127.0.0.1:3000');
const model = process.env.EXPECTED_VLM_MODEL ?? 'qwen3-vl-32b';
const fixturePath = process.env.WEBUI_VLM_FIXTURE ?? '/fixtures/red.png';
// Reuse the pinned base-stack browser account so the GPU overlay works both
// with a fresh WebUI volume and with one created by webui_browser_smoke.mjs.
const email = process.env.WEBUI_VLM_EMAIL ?? 'browser-smoke@kairyu.local';
const password = process.env.WEBUI_VLM_PASSWORD ?? 'KairyuBrowserSmoke-197!';
const fullName = process.env.WEBUI_VLM_NAME ?? 'Kairyu Browser Smoke';
const prompt =
	'Read the single large word shown in the image. Reply with exactly that uppercase word, RED or BLUE, and no other text.';
const actionTimeoutMs = positiveIntegerEnv('WEBUI_VLM_ACTION_TIMEOUT_MS', 30_000);
const navigationTimeoutMs = positiveIntegerEnv('WEBUI_VLM_NAVIGATION_TIMEOUT_MS', 60_000);
const responseTimeoutMs = positiveIntegerEnv('WEBUI_VLM_RESPONSE_TIMEOUT_MS', 300_000);

let browser;
let context;
let page;
let currentStep = 'startup';

function positiveIntegerEnv(name, fallback) {
	const raw = process.env[name];
	if (raw === undefined || raw === '') {
		return fallback;
	}
	const value = Number(raw);
	if (!Number.isSafeInteger(value) || value <= 0) {
		throw new Error(`${name} must be a positive integer, got ${raw}`);
	}
	return value;
}

function invariant(condition, message) {
	if (!condition) {
		throw new Error(message);
	}
}

async function step(name, operation) {
	currentStep = name;
	try {
		return await operation();
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		throw new Error(`${name}: ${detail}`, { cause: error });
	}
}

async function becomesVisible(locator, timeoutMs) {
	return locator
		.waitFor({ state: 'visible', timeout: timeoutMs })
		.then(() => true)
		.catch(() => false);
}

function isAllowedBrowserUrl(rawUrl) {
	const url = new URL(rawUrl);
	if (['about:', 'blob:', 'data:'].includes(url.protocol)) {
		return true;
	}
	if (!['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol)) {
		return false;
	}
	return url.host === baseUrl.host;
}

async function goto(pathname) {
	await page.goto(new URL(pathname, baseUrl).href, {
		waitUntil: 'domcontentloaded',
		timeout: navigationTimeoutMs
	});
}

async function waitForChatSurface() {
	await page.locator('#model-selector-model-button').waitFor({
		state: 'visible',
		timeout: navigationTimeoutMs
	});
	await page.locator('#chat-input').waitFor({
		state: 'visible',
		timeout: navigationTimeoutMs
	});
}

async function dismissFreshInstallChangelog() {
	const accept = page.getByRole('button', { name: "Okay, Let's Go!", exact: true });
	if (await becomesVisible(accept, 5_000)) {
		await accept.click({ timeout: actionTimeoutMs });
	}
}

async function authenticate() {
	await goto('/auth');
	const getStarted = page.getByRole('button', { name: 'Get started', exact: true });
	if (await becomesVisible(getStarted, 5_000)) {
		await getStarted.click({ timeout: actionTimeoutMs });
	}

	await page.locator('#auth-login-card').waitFor({
		state: 'visible',
		timeout: navigationTimeoutMs
	});
	const nameInput = page.locator('#name');
	if (await becomesVisible(nameInput, 2_000)) {
		await nameInput.fill(fullName);
		await page.locator('#email').fill(email);
		await page.locator('#password').fill(password);
		const confirmation = page.locator('#confirm-password');
		if (await becomesVisible(confirmation, 1_000)) {
			await confirmation.fill(password);
		}
		await page
			.getByRole('button', { name: /^(Create Admin Account|Create Account)$/, exact: true })
			.click({ timeout: actionTimeoutMs });
		await dismissFreshInstallChangelog();
	} else {
		await page.locator('#email').fill(email);
		await page.locator('#password').fill(password);
		await page.getByRole('button', { name: 'Sign in', exact: true }).click({
			timeout: actionTimeoutMs
		});
	}
	await waitForChatSurface();
	const authenticated = await page.evaluate(() => Boolean(localStorage.token));
	invariant(authenticated, 'chat surface has no authenticated browser session');
}

async function selectModel() {
	const trigger = page.locator('#model-selector-model-button');
	await trigger.click({ timeout: actionTimeoutMs });
	const search = page.locator('#model-search-input');
	await search.waitFor({ state: 'visible', timeout: actionTimeoutMs });
	await search.fill(model);
	const option = page.locator(`[role="option"][data-value="${model}"]`);
	await option.waitFor({ state: 'visible', timeout: actionTimeoutMs });
	await option.click({ timeout: actionTimeoutMs });
	await page.waitForFunction(
		(expected) =>
			document
				.querySelector('#model-selector-model-button')
				?.getAttribute('aria-label')
				?.includes(expected),
		model,
		{ timeout: actionTimeoutMs }
	);
}

async function locateImageFileInput() {
	let inputs = page.locator('input[type="file"]');
	if ((await inputs.count()) === 0) {
		const chatForm = page.locator('form:has(#chat-input)');
		const attach = chatForm.getByRole('button', {
			name: /attach|upload|add files|more/i
		});
		if ((await attach.count()) > 0) {
			await attach.first().click({ timeout: actionTimeoutMs });
		}
		inputs = page.locator('input[type="file"]');
		await inputs.first().waitFor({ state: 'attached', timeout: actionTimeoutMs });
	}

	const descriptions = [];
	let bestIndex = -1;
	let bestScore = -1;
	for (let index = 0; index < (await inputs.count()); index += 1) {
		const input = inputs.nth(index);
		const descriptor = await input.evaluate((element) => ({
			id: element.id,
			name: element.getAttribute('name') ?? '',
			accept: element.getAttribute('accept') ?? '',
			visible: Boolean(element.offsetWidth || element.offsetHeight || element.getClientRects().length)
		}));
		descriptions.push(descriptor);
		const identity = `${descriptor.id} ${descriptor.name}`.toLowerCase();
		const acceptsImage = descriptor.accept.toLowerCase().includes('image');
		const score = (acceptsImage ? 8 : 0) + (descriptor.visible ? 4 : 0) +
			(/file|upload|attach/.test(identity) ? 2 : 0);
		if (score > bestScore) {
			bestScore = score;
			bestIndex = index;
		}
	}
	invariant(bestIndex >= 0, `no file input was found: ${JSON.stringify(descriptions)}`);
	const selected = inputs.nth(bestIndex);
	const accept = (await selected.getAttribute('accept')) ?? '';
	invariant(
		accept === '' || accept.toLowerCase().includes('image') || accept.includes('*/*'),
		`best file input does not accept images: ${JSON.stringify(descriptions)}`
	);
	return { selected, descriptions };
}

function assertRedAnswer(text) {
	const words = text.toUpperCase().match(/[A-Z]+/g) ?? [];
	invariant(words.includes('RED'), `VLM response did not identify RED: ${text}`);
	invariant(!words.includes('BLUE'), `VLM response was ambiguous: ${text}`);
}

async function uploadAndChat() {
	await selectModel();
	const { selected: fileInput, descriptions } = await locateImageFileInput();
	const uploadResponsePromise = page.waitForResponse(
		(response) => {
			const request = response.request();
			return (
				new URL(response.url()).pathname === '/api/v1/files/' &&
				request.method() === 'POST'
			);
		},
		{ timeout: responseTimeoutMs }
	);
	await fileInput.setInputFiles(fixturePath);
	const uploadResponse = await uploadResponsePromise;
	invariant(
		uploadResponse.status() >= 200 && uploadResponse.status() < 300,
		`Open WebUI image upload returned HTTP ${uploadResponse.status()}`
	);
	const upload = await uploadResponse.json();
	invariant(
		typeof upload?.id === 'string' && upload.id.length > 0,
		`Open WebUI image upload returned no file id: ${JSON.stringify(upload)}`
	);
	const uploadRequestText = uploadResponse.request().postData() ?? '';
	invariant(
		uploadRequestText.includes('filename="red.png"') &&
			uploadRequestText.includes('image/png'),
		'file input upload did not carry the selected PNG multipart metadata'
	);
	const uploadedContentPath = `/api/v1/files/${encodeURIComponent(upload.id)}/content`;
	await page.waitForFunction(
		(expectedPath) =>
			Array.from(document.images).some((image) => {
				try {
					return new URL(image.src, window.location.href).pathname === expectedPath;
				} catch {
					return false;
				}
			}),
		uploadedContentPath,
		{ timeout: responseTimeoutMs }
	);

	const log = page.locator('ul[role="log"]');
	const before = await log.locator('[role="listitem"]').count();
	await page.locator('#chat-input').fill(prompt);
	const responsePromise = page.waitForResponse(
		(response) => {
			const request = response.request();
			if (
				new URL(response.url()).pathname !== '/api/chat/completions' ||
				request.method() !== 'POST'
			) {
				return false;
			}
			try {
				const body = request.postDataJSON();
				return body?.model === model && JSON.stringify(body).includes(prompt);
			} catch {
				return false;
			}
		},
		{ timeout: responseTimeoutMs }
	);
	await page.locator('#send-message-button').click({ timeout: actionTimeoutMs });
	const networkResponse = await responsePromise;
	const requestBody = networkResponse.request().postDataJSON();
	const serializedRequest = JSON.stringify(requestBody);
	invariant(requestBody?.model === model, `UI selected unexpected model ${requestBody?.model}`);
	invariant(requestBody?.stream === true, 'Open WebUI UI image chat was not streaming');
	const attachedFiles = requestBody?.user_message?.files;
	invariant(
		Array.isArray(attachedFiles) && attachedFiles.length === 1,
		`Open WebUI chat request did not carry one uploaded file reference: ${serializedRequest}`
	);
	const attachedFile = attachedFiles[0];
	invariant(
		attachedFile?.id === upload.id &&
			attachedFile?.type === 'file' &&
			attachedFile?.content_type === 'image/png' &&
			attachedFile?.url === upload.id,
		`Open WebUI chat file reference did not preserve uploaded id/type/url: ${JSON.stringify(attachedFile)}`
	);
	invariant(
		!serializedRequest.includes('data:image/png;base64,'),
		'normal Open WebUI chat unexpectedly bypassed its owned-file upload path'
	);
	invariant(networkResponse.status() === 200, `UI image chat returned HTTP ${networkResponse.status()}`);

	await page.waitForFunction(
		(expectedCount) =>
			document.querySelectorAll('ul[role="log"] [role="listitem"]').length >= expectedCount,
		before + 2,
		{ timeout: responseTimeoutMs }
	);
	const responseItem = log.locator('[role="listitem"]').nth(before + 1);
	await responseItem.locator('.copy-response-button').waitFor({
		state: 'visible',
		timeout: responseTimeoutMs
	});
	const answer = (await responseItem.innerText()).trim();
	invariant(answer.length > 0, 'Open WebUI rendered an empty assistant answer');
	invariant(
		!/error|server|connection|provider|upstream|not found|502/i.test(answer),
		`Open WebUI rendered an upstream error: ${answer}`
	);
	assertRedAnswer(answer);
	return {
		answer,
		file: {
			id: upload.id,
			contentType: 'image/png',
			contentPath: uploadedContentPath
		},
		chatRequestBytes: serializedRequest.length,
		fileInputs: descriptions
	};
}

try {
	browser = await chromium.launch({ headless: true });
	context = await browser.newContext();
	await context.route('**/*', async (route) => {
		if (isAllowedBrowserUrl(route.request().url())) {
			await route.continue();
		} else {
			await route.abort('blockedbyclient');
		}
	});
	page = await context.newPage();
	await step('authenticate with pinned OpenWebUI', authenticate);
	const result = await step('real file-input image chat', uploadAndChat);
	console.log(JSON.stringify({ model, passed: true, ...result }));
} catch (error) {
	const detail = error instanceof Error ? error.stack ?? error.message : String(error);
	console.error(`WEBUI VLM BROWSER FAIL at ${currentStep}: ${detail}`);
	process.exitCode = 1;
} finally {
	await context?.close().catch(() => {});
	await browser?.close().catch(() => {});
}
