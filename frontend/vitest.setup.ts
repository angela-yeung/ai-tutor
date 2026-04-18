import "@testing-library/jest-dom/vitest";

// Set up environment variables for tests
process.env.NEXT_PUBLIC_API_URL = "http://localhost:3000/api";

// Ensure localStorage has all required methods (jsdom doesn't always provide them)
if (typeof window !== "undefined") {
  const store: Record<string, string> = {};

  // Create a proper storage object with all methods
  const storageProxy = {
    getItem: (key: string) => (key in store ? store[key] : null),
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      for (const key in store) {
        delete store[key];
      }
    },
    key: (index: number) => {
      const keys = Object.keys(store);
      return keys[index] || null;
    },
    get length() {
      return Object.keys(store).length;
    },
  };

  // Replace window.localStorage with our proxy
  Object.defineProperty(window, "localStorage", {
    value: storageProxy,
    writable: true,
    enumerable: true,
    configurable: true,
  });
}
