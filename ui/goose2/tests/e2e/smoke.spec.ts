import { expect, test } from "@playwright/test";
import {
  expect as tauriExpect,
  test as tauriTest,
  waitForHome,
} from "./fixtures/tauri-mock";

test.describe("Smoke tests", () => {
  test("app loads and shows home screen", async ({ page }) => {
    await page.goto("/");

    // Wait for the app to render — greeting should appear
    await expect(
      page.getByText(/Good (morning|afternoon|evening)/),
    ).toBeVisible({ timeout: 10_000 });
  });

  test("home screen shows clock", async ({ page }) => {
    await page.goto("/");

    // Should show AM or PM once the clock renders
    await expect(page.getByText(/[AP]M/)).toBeVisible({ timeout: 10_000 });
  });

  test("home screen shows chat input placeholder", async ({ page }) => {
    await page.goto("/");

    await expect(
      page.getByPlaceholder(/Message .*, @ to mention agents/),
    ).toBeVisible({ timeout: 10_000 });
  });
});

tauriTest.describe("Appearance settings", () => {
  tauriTest(
    "uses system by default, then applies a theme and accent vars together",
    async ({ tauriMocked: page }) => {
      await page.goto("/");
      await waitForHome(page);

      await page.evaluate(() => {
        window.dispatchEvent(
          new CustomEvent("goose:open-settings", {
            detail: { section: "appearance" },
          }),
        );
      });

      await tauriExpect(page.getByTestId("theme-option-system")).toBeVisible();

      await tauriExpect
        .poll(() =>
          page.evaluate(() => ({
            customTheme: window.localStorage.getItem("goose-custom-theme"),
            htmlClass: document.documentElement.className,
          })),
        )
        .toEqual({
          customTheme: null,
          htmlClass: expect.stringMatching(/light|dark/),
        });

      await tauriExpect(page.getByTestId("accent-color-red")).toBeDisabled();

      await page.getByTestId("theme-search-input").fill("github-light");
      await page.getByTestId("theme-option-github-light").click();
      await tauriExpect(page.locator("html")).toHaveClass(/light/);
      await tauriExpect(page.getByTestId("accent-color-red")).toBeEnabled();

      await page.getByTestId("theme-search-input").fill("dracula");
      await page.getByTestId("theme-option-dracula").click();
      await tauriExpect(page.locator("html")).toHaveClass(/dark/);

      await page.getByTestId("accent-color-red").click();

      await tauriExpect
        .poll(() =>
          page.evaluate(() => ({
            customTheme: window.localStorage.getItem("goose-custom-theme"),
            accent: window.localStorage.getItem("goose-accent-color"),
            background: getComputedStyle(document.documentElement)
              .getPropertyValue("--background")
              .trim(),
            primary: getComputedStyle(document.documentElement)
              .getPropertyValue("--primary")
              .trim(),
            sidebar: getComputedStyle(document.documentElement)
              .getPropertyValue("--sidebar-background")
              .trim(),
            warning: getComputedStyle(document.documentElement)
              .getPropertyValue("--ui-warning")
              .trim(),
            brand: getComputedStyle(document.documentElement)
              .getPropertyValue("--brand-color")
              .trim(),
          })),
        )
        .toEqual({
          customTheme: "dracula",
          accent: "#ef4444",
          background: expect.any(String),
          primary: expect.any(String),
          sidebar: expect.any(String),
          warning: expect.any(String),
          brand: "#ef4444",
        });

      await page.getByTestId("theme-option-system").click();

      await tauriExpect(page.getByTestId("accent-color-red")).toBeDisabled();
      await tauriExpect
        .poll(() =>
          page.evaluate(() =>
            window.localStorage.getItem("goose-custom-theme"),
          ),
        )
        .toBeNull();

      const systemColors = await page.evaluate(() => ({
        primary: getComputedStyle(document.documentElement)
          .getPropertyValue("--primary")
          .trim(),
        brand: getComputedStyle(document.documentElement)
          .getPropertyValue("--brand-color")
          .trim(),
      }));

      expect(systemColors.primary).not.toBe("0 84.2% 60.2%");
      expect(systemColors.brand).not.toBe("#ef4444");
    },
  );
});
