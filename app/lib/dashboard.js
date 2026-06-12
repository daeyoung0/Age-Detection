export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const MENU_ITEMS = [
  { name: "아이스 아메리카노", price: 2000, category: "커피", brand: "Vision Cafe" },
  { name: "핫 아메리카노", price: 2000, category: "커피", brand: "Vision Cafe" },
  { name: "카페라떼", price: 5000, category: "커피", brand: "Vision Cafe" },
  { name: "바닐라라떼", price: 5500, category: "커피", brand: "Vision Cafe" },
  { name: "딸기 스무디", price: 5800, category: "음료", brand: "Vision Cafe" },
  { name: "망고 에이드", price: 5500, category: "음료", brand: "Vision Cafe" },
  { name: "치즈 케이크", price: 6000, category: "디저트", brand: "Vision Bakery" },
  { name: "초코 브라우니", price: 5200, category: "디저트", brand: "Vision Bakery" },
  { name: "크로플", price: 4800, category: "디저트", brand: "Vision Bakery" },
];

export const PAYMENT_LABELS = {
  cash: "현금",
  card: "카드",
};

export function formatCurrency(value) {
  return `${Number(value ?? 0).toLocaleString("ko-KR")}원`;
}

export function formatGender(gender) {
  if (gender === "male") return "남성";
  if (gender === "female") return "여성";
  return "미분류";
}

export async function parseJsonResponse(response) {
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchDashboardData() {
  const [statsData, usersData, productsData, transactionsData] = await Promise.all([
    fetch(`${BACKEND_URL}/api/stats`, { cache: "no-store" }).then(parseJsonResponse),
    fetch(`${BACKEND_URL}/api/users`, { cache: "no-store" }).then(parseJsonResponse),
    fetch(`${BACKEND_URL}/api/products`, { cache: "no-store" }).then(parseJsonResponse),
    fetch(`${BACKEND_URL}/api/transactions`, { cache: "no-store" }).then(parseJsonResponse),
  ]);

  return {
    stats: statsData.statistics ?? [],
    users: usersData.users ?? [],
    products: productsData.products ?? [],
    transactions: transactionsData.transactions ?? [],
  };
}
