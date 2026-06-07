import Link from "next/link";

const cards = [
  {
    href: "/pos",
    eyebrow: "Checkout",
    title: "POS 결제 화면",
    description: "상품 선택, 결제 수단 입력, 최근 결제 이력을 한 화면에서 처리합니다.",
  },
  {
    href: "/stats",
    eyebrow: "Analytics",
    title: "통계 분석 화면",
    description: "연령대·성별 매출, 최근 고객, 상품, 거래 이력을 분리해서 확인합니다.",
  },
];

export default function HomePage() {
  return (
    <main className="page-shell">
      <section className="hero hero-home">
        <div className="hero-copy-block">
          <p className="eyebrow">AI Consumer Analysis</p>
          <h1>소비자 분석 시스템 대시보드</h1>
          <p className="hero-copy">
            카메라 기반 연령·성별 추정 결과와 POS 구매 이력을 연결해 매장 소비 패턴을 분석하는
            운영용 화면입니다.
          </p>
        </div>
        <div className="hero-panel">
          <p>바로가기</p>
          <div className="launch-grid">
            {cards.map((card) => (
              <Link key={card.href} href={card.href} className="launch-card">
                <span>{card.eyebrow}</span>
                <strong>{card.title}</strong>
                <p>{card.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
