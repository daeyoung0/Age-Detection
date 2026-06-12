"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "홈" },
  { href: "/pos", label: "POS" },
  { href: "/stats", label: "통계" },
];

export default function TopNav() {
  const pathname = usePathname();

  return (
    <header className="top-nav-shell">
      <div className="top-nav">
        <Link href="/" className="brand-mark">
          <span className="brand-dot" />
          <div>
            <strong>Vision Retail Lab</strong>
            <p>Consumer Analysis Suite</p>
          </div>
        </Link>
        <nav className="nav-links">
          {items.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={pathname === item.href ? "nav-link active" : "nav-link"}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
