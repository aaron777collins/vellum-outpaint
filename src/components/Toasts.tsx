import { useStore } from "../store";

export default function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismiss);
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} onClick={() => dismiss(t.id)}>
          <span className="toast-dot" />
          <span className="toast-msg">{t.msg}</span>
        </div>
      ))}
    </div>
  );
}
