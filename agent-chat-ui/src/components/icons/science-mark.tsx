import { Microscope } from "lucide-react";

export function ScienceMark() {
  return (
    <span
      aria-hidden="true"
      className="bg-primary text-primary-foreground grid size-8 shrink-0 place-items-center rounded-lg"
    >
      <Microscope className="size-5" />
    </span>
  );
}
