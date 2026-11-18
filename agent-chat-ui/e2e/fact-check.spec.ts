import { expect, test } from "@playwright/test";

test("streams full-text progress, renders a PDF citation, and retries the failed claim", async ({
  page,
}) => {
  await page.goto("/");
  const composer = page.getByPlaceholder(
    "Ask a research question or verify a claim…",
  );
  await composer.fill("Fact-check both fixture claims.");
  await composer.press("Enter");

  await expect(
    page.getByRole("status").filter({ hasText: "Checking full-text passages" }),
  ).toBeVisible({ timeout: 20_000 });

  const report = page
    .getByRole("region", { name: "Fact-check report" })
    .first();
  await expect(
    report.getByRole("heading", { name: "Supported", exact: true }),
  ).toBeVisible();
  await expect(
    report.getByRole("heading", { name: "Verification unavailable" }),
  ).toBeVisible();
  await report.getByText("Supporting evidence", { exact: false }).click();
  await expect(
    report.getByText("The coated panel was 8 °C cooler than the control."),
  ).toBeVisible();
  await expect(
    report.getByRole("link", { name: "PDF page 4" }),
  ).toHaveAttribute("href", "https://example.org/coating-study.pdf#page=4");

  const retry = report.getByRole("button", { name: "Retry this claim" });
  await expect(retry).toBeEnabled();
  await retry.click();
  await expect(
    page.getByText(
      "Fact-check this claim: The backup sensor maintained its calibration.",
      { exact: true },
    ),
  ).toBeVisible();
});
