"use client";

import { useEffect, useMemo, useState } from "react";
import {
  fetchDashboardData,
  formatCurrency,
  formatGender,
  PAYMENT_LABELS,
} from "../lib/dashboard";

function groupByCategory(products, transactions) {
  const productMap = new Map(products.map((product) => [product.product_id, product]));
  const totals = new Map();
  for (const tx of transactions) {
    const product = productMap.get(tx.product_id);
    const category = product?.category || "기타";
    totals.set(category, (totals.get(category) || 0) + Number(tx.total_amount || 0));
  }
  return Array.from(totals.entries())
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
}

function groupByPayment(transactions) {
  const totals = new Map();
  for (const tx of transactions) {
    const label = PAYMENT_LABELS[tx.payment_method] || tx.payment_method || "기타";
    totals.set(label, (totals.get(label) || 0) + Number(tx.total_amount || 0));
  }
  return Array.from(totals.entries())
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
}

function groupHourly(transactions) {
  const hours = Array.from({ length: 24 }, (_, hour) => ({ hour, value: 0 }));
  for (const tx of transactions) {
    const date = new Date(tx.purchase_time);
    const hour = Number.isNaN(date.getHours()) ? 0 : date.getHours();
    hours[hour].value += Number(tx.total_amount || 0);
  }
  return hours;
}

function buildTrend(transactions) {
  const totals = new Map();
  for (const tx of transactions) {
    const date = new Date(tx.purchase_time);
    const key = Number.isNaN(date.getTime())
      ? "Unknown"
      : `${String(date.getMonth() + 1).padStart(2, "0")}.${String(date.getDate()).padStart(2, "0")}`;
    totals.set(key, (totals.get(key) || 0) + Number(tx.total_amount || 0));
  }
  return Array.from(totals.entries())
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => a.label.localeCompare(b.label))
    .slice(-7);
}

function buildTopProducts(transactions) {
  const totals = new Map();
  for (const tx of transactions) {
    const current = totals.get(tx.product_name) || { label: tx.product_name, count: 0, sales: 0 };
    current.count += Number(tx.quantity || 0);
    current.sales += Number(tx.total_amount || 0);
    totals.set(tx.product_name, current);
  }
  return Array.from(totals.values())
    .sort((a, b) => b.sales - a.sales)
    .slice(0, 5);
}

function buildAgeGenderMatrix(stats) {
  const ages = ["유아/아동 (0-12)", "청소년 (13-18)", "청년층 (19-34)", "중장년층 (35-49)", "고령층 (50+)", "알 수 없음"];
  const genders = ["male", "female", "unknown"];
  const totalSales = stats.reduce((sum, item) => sum + Number(item.total_sales || 0), 0) || 1;
  return genders.map((gender) => ({
    gender,
    values: ages.map((age) => {
      const row = stats.find((item) => item.gender === gender && item.age_group === age);
      const sales = Number(row?.total_sales || 0);
      return Math.round((sales / totalSales) * 100);
    }),
  }));
}

function polarStyle(items, total, colors) {
  let start = 0;
  const segments = items
    .map((item, index) => {
      const angle = total > 0 ? (item.value / total) * 360 : 0;
      const segment = `${colors[index % colors.length]} ${start}deg ${start + angle}deg`;
      start += angle;
      return segment;
    })
    .join(", ");
  return {
    background: `conic-gradient(${segments || "#ddd 0deg 360deg"})`,
  };
}

function LineChart({ points }) {
  const width = 480;
  const height = 180;
  const maxValue = Math.max(...points.map((item) => item.value), 1);
  const coords = points.map((point, index) => {
    const x = (index / Math.max(points.length - 1, 1)) * (width - 24) + 12;
    const y = height - (point.value / maxValue) * (height - 24) - 12;
    return `${x},${y}`;
  });
  return (
    <div className="chart-shell">
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img" aria-label="매출 추이">
        <polyline fill="none" stroke="rgba(82, 140, 255, 0.18)" strokeWidth="16" strokeLinecap="round" strokeLinejoin="round" points={coords.join(" ")} />
        <polyline fill="none" stroke="#3b82f6" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" points={coords.join(" ")} />
        {points.map((point, index) => {
          const [cx, cy] = coords[index].split(",").map(Number);
          return <circle key={point.label} cx={cx} cy={cy} r="5" fill="#3b82f6" />;
        })}
      </svg>
      <div className="chart-labels">
        {points.map((point) => (
          <span key={point.label}>{point.label}</span>
        ))}
      </div>
    </div>
  );
}

export default function StatsPage() {
  const [stats, setStats] = useState([]);
  const [users, setUsers] = useState([]);
  const [products, setProducts] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [statsError, setStatsError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  async function loadStats() {
    try {
      const data = await fetchDashboardData();
      setStats(data.stats);
      setUsers(data.users);
      setProducts(data.products);
      setTransactions(data.transactions);
      setStatsError("");
      setLastUpdated(new Date().toLocaleTimeString("ko-KR"));
    } catch (error) {
      setStatsError("통계 데이터를 불러오지 못했습니다. 백엔드를 확인하세요.");
    }
  }

  useEffect(() => {
    loadStats();
    const timer = setInterval(loadStats, 5000);
    return () => clearInterval(timer);
  }, []);

  const totalSales = useMemo(
    () => transactions.reduce((sum, item) => sum + Number(item.total_amount || 0), 0),
    [transactions]
  );
  const totalOrders = transactions.length;
  const avgOrderValue = totalOrders > 0 ? Math.round(totalSales / totalOrders) : 0;
  const visitors = users.length;

  const categoryData = useMemo(() => groupByCategory(products, transactions), [products, transactions]);
  const paymentData = useMemo(() => groupByPayment(transactions), [transactions]);
  const hourlyData = useMemo(() => groupHourly(transactions), [transactions]);
  const trendData = useMemo(() => buildTrend(transactions), [transactions]);
  const topProducts = useMemo(() => buildTopProducts(transactions), [transactions]);
  const ageGenderMatrix = useMemo(() => buildAgeGenderMatrix(stats), [stats]);

  const categoryTotal = categoryData.reduce((sum, item) => sum + item.value, 0);
  const paymentTotal = paymentData.reduce((sum, item) => sum + item.value, 0);

  const categoryColors = ["#3b82f6", "#2dd4bf", "#8b5cf6", "#f59e0b", "#94a3b8"];
  const paymentColors = ["#3b82f6", "#2dd4bf", "#8b5cf6", "#f59e0b", "#94a3b8"];

  const maxHourly = Math.max(...hourlyData.map((item) => item.value), 1);

  return (
    <main className="page-shell">
      <section className="dashboard-header">
        <div>
          <p className="eyebrow">Statistics Dashboard</p>
          <h1>통계 대시보드</h1>
          <p className="hero-copy">매출 흐름, 카테고리 비중, 결제 수단, 인기 상품, 연령·성별 고객 비율을 한 화면에서 확인합니다.</p>
        </div>
        <div className="dashboard-toolbar">
          <div className="toolbar-chip">실시간 반영</div>
          <div className="toolbar-chip">{lastUpdated || "--:--:--"}</div>
        </div>
      </section>

      {statsError ? <div className="error-banner">{statsError}</div> : null}

      <section className="kpi-grid">
        <article className="kpi-card">
          <span>총 매출</span>
          <strong>{formatCurrency(totalSales)}</strong>
        </article>
        <article className="kpi-card">
          <span>총 주문 건수</span>
          <strong>{totalOrders.toLocaleString("ko-KR")}건</strong>
        </article>
        <article className="kpi-card">
          <span>평균 주문 금액</span>
          <strong>{formatCurrency(avgOrderValue)}</strong>
        </article>
        <article className="kpi-card">
          <span>방문 고객 수</span>
          <strong>{visitors.toLocaleString("ko-KR")}명</strong>
        </article>
      </section>

      <section className="dashboard-grid">
        <article className="dashboard-card span-7">
          <div className="card-title-row">
            <h3>매출 추이</h3>
            <span>최근 7개 구간</span>
          </div>
          {trendData.length === 0 ? <div className="empty-state compact">표시할 데이터가 없습니다.</div> : <LineChart points={trendData} />}
        </article>

        <article className="dashboard-card span-5">
          <div className="card-title-row">
            <h3>상품 카테고리별 매출 비중</h3>
            <span>{categoryData.length} categories</span>
          </div>
          <div className="donut-layout">
            <div className="donut-chart" style={polarStyle(categoryData, categoryTotal, categoryColors)}>
              <div className="donut-hole">
                <strong>{formatCurrency(categoryTotal)}</strong>
                <span>카테고리 합계</span>
              </div>
            </div>
            <div className="legend-list">
              {categoryData.map((item, index) => (
                <div className="legend-row" key={item.label}>
                  <div className="legend-label">
                    <span className="legend-dot" style={{ backgroundColor: categoryColors[index % categoryColors.length] }} />
                    <span>{item.label}</span>
                  </div>
                  <strong>{categoryTotal ? `${((item.value / categoryTotal) * 100).toFixed(1)}%` : "0.0%"}</strong>
                </div>
              ))}
            </div>
          </div>
        </article>

        <article className="dashboard-card span-7">
          <div className="card-title-row">
            <h3>시간대별 매출</h3>
            <span>0시 ~ 23시</span>
          </div>
          <div className="bar-chart-vertical">
            {hourlyData.map((item) => (
              <div className="bar-col" key={item.hour}>
                <div className="bar-col-track">
                  <div className="bar-col-fill" style={{ height: `${(item.value / maxHourly) * 100}%` }} />
                </div>
                <span>{String(item.hour).padStart(2, "0")}시</span>
              </div>
            ))}
          </div>
        </article>

        <article className="dashboard-card span-5">
          <div className="card-title-row">
            <h3>결제 수단 비율</h3>
            <span>{paymentData.length} methods</span>
          </div>
          <div className="donut-layout">
            <div className="donut-chart" style={polarStyle(paymentData, paymentTotal, paymentColors)}>
              <div className="donut-hole">
                <strong>{formatCurrency(paymentTotal)}</strong>
                <span>결제 합계</span>
              </div>
            </div>
            <div className="legend-list">
              {paymentData.map((item, index) => (
                <div className="legend-row" key={item.label}>
                  <div className="legend-label">
                    <span className="legend-dot" style={{ backgroundColor: paymentColors[index % paymentColors.length] }} />
                    <span>{item.label}</span>
                  </div>
                  <strong>{paymentTotal ? `${((item.value / paymentTotal) * 100).toFixed(1)}%` : "0.0%"}</strong>
                </div>
              ))}
            </div>
          </div>
        </article>

        <article className="dashboard-card span-6">
          <div className="card-title-row">
            <h3>인기 상품 TOP 5</h3>
          </div>
          <div className="stats-table top-product-table">
            <div className="table-head top-products">
              <span>순위</span>
              <span>상품명</span>
              <span>판매 수량</span>
              <span>매출액</span>
            </div>
            {topProducts.length === 0 ? (
              <div className="empty-state compact">상품 데이터가 없습니다.</div>
            ) : (
              topProducts.map((item, index) => (
                <div className="table-row top-products" key={item.label}>
                  <span>{index + 1}</span>
                  <span>{item.label}</span>
                  <span>{item.count.toLocaleString("ko-KR")}</span>
                  <span>{formatCurrency(item.sales)}</span>
                </div>
              ))
            )}
          </div>
        </article>

        <article className="dashboard-card span-6">
          <div className="card-title-row">
            <h3>연령/성별 고객 비율</h3>
          </div>
          <div className="stats-table age-gender-table">
            <div className="table-head age-gender">
              <span>성별</span>
              <span>유아/아동</span>
              <span>청소년</span>
              <span>청년층</span>
              <span>중장년층</span>
              <span>고령층</span>
              <span>미상</span>
            </div>
            {ageGenderMatrix.map((row) => (
              <div className="table-row age-gender" key={row.gender}>
                <span>{formatGender(row.gender)}</span>
                {row.values.map((value, index) => (
                  <span key={`${row.gender}-${index}`}>{value}%</span>
                ))}
              </div>
            ))}
          </div>
          <div className="analysis-note">
            분석 인사이트
            <p>
              {stats.length === 0
                ? "아직 누적된 구매 데이터가 없습니다."
                : `${stats[0].age_group} 고객군 비중이 가장 높고, 최근 집계 기준으로 ${paymentData[0]?.label || "결제수단"} 사용 비중이 가장 큽니다.`}
            </p>
          </div>
        </article>
      </section>
    </main>
  );
}
