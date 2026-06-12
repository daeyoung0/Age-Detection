"use client";

import { useMemo, useState } from "react";
import {
  BACKEND_URL,
  MENU_ITEMS,
  PAYMENT_LABELS,
  formatCurrency,
  formatGender,
  parseJsonResponse,
} from "../lib/dashboard";

const DEFAULT_STATUS = "결제 대기 중";
const CATEGORY_ORDER = ["커피", "디저트", "주식"];

export default function PosPage() {
  const [selectedItem, setSelectedItem] = useState(null);
  const [quantity, setQuantity] = useState(1);
  const [isPaying, setIsPaying] = useState(false);
  const [statusText, setStatusText] = useState(DEFAULT_STATUS);
  const [history, setHistory] = useState([]);
  const [errorText, setErrorText] = useState("");

  const totalAmount = useMemo(() => {
    return (selectedItem?.price ?? 0) * quantity;
  }, [selectedItem, quantity]);

  function selectItem(item) {
    setSelectedItem(item);
    setQuantity(1);
    setStatusText(`${item.name} 선택됨`);
    setErrorText("");
  }

  function changeQuantity(delta) {
    setQuantity((current) => {
      const next = current + delta;
      return Math.max(1, Math.min(99, next));
    });
  }

  function handleQuantityInput(event) {
    const next = Number(event.target.value);
    if (Number.isNaN(next)) {
      setQuantity(1);
      return;
    }
    setQuantity(Math.max(1, Math.min(99, next)));
  }

  async function handlePayment(paymentMethod) {
    if (!selectedItem || isPaying) {
      return;
    }

    setIsPaying(true);
    setStatusText(`${PAYMENT_LABELS[paymentMethod]} 결제 처리 중`);
    setErrorText("");

    try {
      const payload = {
        item_name: selectedItem.name,
        price: selectedItem.price,
        quantity,
        payment_method: paymentMethod,
        category: selectedItem.category,
        brand: selectedItem.brand,
      };

      const data = await fetch(`${BACKEND_URL}/api/pay`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then(parseJsonResponse);

      setHistory((prev) => [
        {
          id: data.transaction_id ?? Date.now(),
          item: data.item,
          price: selectedItem.price,
          quantity,
          totalAmount,
          paymentMethod,
          ageGroup: data.matched_age_group,
          gender: data.matched_gender,
          age: data.matched_age,
          createdAt: new Date().toLocaleTimeString("ko-KR"),
        },
        ...prev,
      ]);
      setSelectedItem(null);
      setQuantity(1);
      setStatusText("결제 완료");
    } catch (error) {
      setStatusText("결제 실패");
      setErrorText("결제 처리에 실패했습니다. 백엔드와 카메라 로그를 확인하세요.");
    } finally {
      setIsPaying(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero hero-compact">
        <div className="hero-copy-block">
          <p className="eyebrow">POS Workspace</p>
          <h1>매장 결제 화면</h1>
          <p className="hero-copy">
            상품을 고르고 수량을 선택한 뒤 결제를 진행합니다. 결제 결과는 최근 고객 추정 정보와 함께 저장됩니다.
          </p>
        </div>
        <div className="hero-panel status-panel">
          <p>현재 상태</p>
          <strong>{statusText}</strong>
          <span>{errorText || "카메라 로그가 연결되면 고객 추정 정보가 자동으로 함께 기록됩니다."}</span>
        </div>
      </section>

      <section className="workspace">
        <div className="pos-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Order Desk</p>
              <h2>상품 선택</h2>
            </div>
            <span className="status-pill">{selectedItem ? selectedItem.category : "미선택"}</span>
          </div>

          <div className="menu-groups">
            {CATEGORY_ORDER.map((category) => (
              <div key={category} className="menu-group">
                <h3>{category}</h3>
                <div className="menu-grid">
                  {MENU_ITEMS.filter((item) => item.category === category).map((item) => (
                    <button
                      key={item.name}
                      type="button"
                      className={selectedItem?.name === item.name ? "menu-card selected" : "menu-card"}
                      onClick={() => selectItem(item)}
                    >
                      <span>{item.name}</span>
                      <small>{item.brand}</small>
                      <strong>{formatCurrency(item.price)}</strong>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="checkout-card">
            <div className="checkout-row">
              <span>선택 상품</span>
              <strong>{selectedItem?.name ?? "없음"}</strong>
            </div>
            <div className="checkout-row">
              <span>브랜드</span>
              <strong>{selectedItem?.brand ?? "-"}</strong>
            </div>
            <div className="checkout-row">
              <span>카테고리</span>
              <strong>{selectedItem?.category ?? "-"}</strong>
            </div>
            <div className="quantity-row">
              <span>수량</span>
              <div className="quantity-control">
                <button
                  type="button"
                  className="quantity-button"
                  onClick={() => changeQuantity(-1)}
                  disabled={!selectedItem || isPaying || quantity <= 1}
                >
                  -
                </button>
                <input
                  className="quantity-input"
                  type="number"
                  min="1"
                  max="99"
                  value={quantity}
                  onChange={handleQuantityInput}
                  disabled={!selectedItem || isPaying}
                />
                <button
                  type="button"
                  className="quantity-button"
                  onClick={() => changeQuantity(1)}
                  disabled={!selectedItem || isPaying || quantity >= 99}
                >
                  +
                </button>
              </div>
            </div>
            <div className="checkout-row total">
              <span>결제 금액</span>
              <strong>{formatCurrency(totalAmount)}</strong>
            </div>
            <div className="action-row">
              <button type="button" disabled={!selectedItem || isPaying} onClick={() => handlePayment("cash")}>
                현금 결제
              </button>
              <button type="button" disabled={!selectedItem || isPaying} onClick={() => handlePayment("card")}>
                카드 결제
              </button>
            </div>
          </div>
        </div>

        <aside className="history-panel">
          <div className="panel-header slim">
            <div>
              <p className="panel-kicker">Recent Orders</p>
              <h2>최근 결제</h2>
            </div>
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <div className="empty-state">최근 결제 내역이 없습니다.</div>
            ) : (
              history.map((item) => (
                <article key={item.id} className="history-item">
                  <div className="history-top">
                    <strong>{item.item}</strong>
                    <span>{formatCurrency(item.totalAmount ?? item.price)}</span>
                  </div>
                  <p>{item.createdAt}</p>
                  <p>수량: {item.quantity}</p>
                  <p>연령대: {item.ageGroup}</p>
                  <p>성별: {formatGender(item.gender)}</p>
                  <p>결제: {PAYMENT_LABELS[item.paymentMethod] ?? item.paymentMethod}</p>
                </article>
              ))
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
