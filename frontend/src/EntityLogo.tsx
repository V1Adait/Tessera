import { useState } from "react";
import { TONE_CLASSES } from "@/components/ui/tones";
import { cn } from "@/lib/utils";
import { LOGO_TEMPLATE, avatarTone, initialsOf, logoUrl } from "./logoModel";

interface Props {
  name: string;
  domain?: string | null;
  showLogos: boolean; // off by default: a logo lookup tells an outside service which companies you research
  className?: string;
}

/** A company's avatar: its logo when logos are on and it has a website, otherwise its initials in a
 * colour that is always the same for that company. */
export default function EntityLogo({ name, domain, showLogos, className }: Props) {
  // remember which domain failed to load, so a changed domain gets a fresh try
  const [failed, setFailed] = useState<string | null>(null);
  const url = showLogos && domain && failed !== domain ? logoUrl(LOGO_TEMPLATE, domain) : null;

  if (url !== null) {
    return (
      <img
        src={url}
        alt=""
        loading="lazy"
        referrerPolicy="no-referrer" // do not tell the icon service which app, or page, asked
        onError={() => setFailed(domain ?? null)}
        className={cn("size-5 shrink-0 rounded-sm border bg-background object-contain", className)}
      />
    );
  }
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-5 shrink-0 items-center justify-center rounded-sm text-[9px] font-semibold leading-none",
        TONE_CLASSES[avatarTone(name)],
        className,
      )}
    >
      {initialsOf(name)}
    </span>
  );
}
