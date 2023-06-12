interface LayoutProps {
  children?: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
<div className="main-bg" style={{ minHeight: '100vh', background: 'linear-gradient(135deg, rgba(90, 200, 254, 0.1), rgba(19, 202, 152, 0.2))' }}>
    <div className="mx-auto flex flex-col space-y-4">
      <header className="container sticky top-0 z-40 bg-blue">
          <nav className="ml-4 pl-6">
          </nav>
      </header>
      <div>
        <main className="flex w-full flex-1 flex-col overflow-hidden">
          {children}
        </main>
      </div>
    </div>
</div>
  );
}
