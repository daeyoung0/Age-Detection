import "./globals.css";
import TopNav from "./components/top-nav";

export const metadata = {
  title: "Consumer Analysis Dashboard",
  description: "POS and consumer analytics dashboard powered by age and gender estimation",
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body>
        <TopNav />
        {children}
      </body>
    </html>
  );
}
