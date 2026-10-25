import { Hotel, Plane, Send } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { clock, cn } from "@/lib/format";
import { useConsole, useNow } from "@/lib/useConsole";
import { Button, Card, Empty, Field, PageHeader, inputClass } from "@/components/ui";
import { Pending } from "@/components/Pending";

// Flights and hotels against a mock endpoint (FR-54, FR-55). A request from
// here takes the same path as one made by voice: the executor holds it, and
// nothing is booked until the occupant says yes.

function NewBooking() {
  const [kind, setKind] = useState<"flight" | "hotel">("flight");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [day, setDay] = useState("");
  const [nights, setNights] = useState(2);
  const [busy, setBusy] = useState(false);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      await api.book({ kind, destination, depart_on: `${day}T09:00:00`, origin: kind === "flight" ? origin : "", nights: kind === "hotel" ? nights : null });
      toast.success("Requested; confirm it above when it appears");
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="grid gap-3">
      <div className="grid grid-cols-2 gap-1 rounded-xl bg-raised/60 p-1" role="radiogroup" aria-label="What to book">
        {(["flight", "hotel"] as const).map((value) => {
          const Icon = value === "flight" ? Plane : Hotel;
          return (
            <button key={value} type="button" role="radio" aria-checked={kind === value} onClick={() => setKind(value)}
              className={cn("flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-[13px] capitalize", kind === value ? "bg-surface text-fg shadow" : "text-muted hover:text-fg")}>
              <Icon className="h-3.5 w-3.5" />{value}
            </button>
          );
        })}
      </div>
      {kind === "flight" && <Field label="From"><input id="booking-origin" value={origin} onChange={(event) => setOrigin(event.target.value)} placeholder="Hyderabad" className={inputClass} /></Field>}
      <Field label={kind === "flight" ? "To" : "Where"}><input id="booking-destination" required value={destination} onChange={(event) => setDestination(event.target.value)} placeholder={kind === "flight" ? "Goa" : "Pune"} className={inputClass} /></Field>
      <div className={cn("grid gap-3", kind === "hotel" && "grid-cols-2")}>
        <Field label={kind === "flight" ? "Date" : "Check in"}><input id="booking-day" type="date" required value={day} onChange={(event) => setDay(event.target.value)} className={inputClass} /></Field>
        {kind === "hotel" && <Field label="Nights"><input id="booking-nights" type="number" min={1} max={60} value={nights} onChange={(event) => setNights(Number(event.target.value))} className={inputClass} /></Field>}
      </div>
      <Button type="submit" variant="primary" icon={Send} disabled={busy || !destination.trim() || !day}>Request booking</Button>
      <p className="text-[11.5px] text-muted">Nothing is booked until you confirm it. Every booking reaches a mock endpoint; no real reservation is ever made.</p>
    </form>
  );
}

export function Bookings() {
  const { state } = useConsole();
  const now = useNow();
  if (!state) return null;
  const waiting = state.waiting.filter((item) => item.tool === "book_travel");
  const history = [...state.bookings].reverse();
  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader title="Bookings" subtitle="Flights and hotels, by voice, chat or the form here. Each one waits for your confirmation and reaches a mock endpoint: it is a demonstration of the confirmation gate, not a travel agent." />
      <div className="grid gap-4">
        <Pending items={waiting} now={now} noun="booking" />
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Card title="Made" icon={Plane} padded={false}>
            {history.length === 0 ? (
              <Empty icon={Plane} title="No bookings yet">Ask Jarvis to book a flight or a hotel, or use the form. Confirmed bookings are listed here with their reference.</Empty>
            ) : (
              <div className="scroll-thin overflow-x-auto">
                <table className="w-full min-w-[640px] text-[13px]">
                  <thead>
                    <tr className="text-left text-[11px] uppercase tracking-[0.06em] text-muted">
                      <th className="px-5 py-2.5 font-medium">Reference</th>
                      <th className="px-3 py-2.5 font-medium">What</th>
                      <th className="px-3 py-2.5 font-medium">When</th>
                      <th className="px-3 py-2.5 font-medium">Booked</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((booking) => (
                      <tr key={`${booking.reference}-${booking.ts}`} className="border-t hairline">
                        <td className="px-5 py-2.5 font-mono text-glacier">{booking.reference}</td>
                        <td className="px-3 py-2.5 text-fg">
                          <span className="flex items-center gap-2">
                            {booking.kind === "hotel" ? <Hotel className="h-3.5 w-3.5 text-muted" /> : <Plane className="h-3.5 w-3.5 text-muted" />}
                            {booking.kind === "hotel" ? `Hotel in ${booking.destination}${booking.nights ? `, ${booking.nights} nights` : ""}` : `${booking.origin ? `${booking.origin} → ` : ""}${booking.destination}`}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 font-mono text-fg/85">{booking.depart_on.slice(0, 10)}</td>
                        <td className="px-3 py-2.5 text-muted">{clock(booking.ts)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
          <Card title="New booking" icon={Send}><NewBooking /></Card>
        </div>
      </div>
    </div>
  );
}
