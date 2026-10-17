import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const button = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-sm transition-colors " +
    "focus-visible:outline-2 focus-visible:outline-ring disabled:pointer-events-none disabled:opacity-50 cursor-pointer",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        outline: "border bg-background hover:bg-muted",
        ghost: "hover:bg-muted",
        link: "text-primary hover:underline px-0",
      },
      size: { default: "h-8 px-3", sm: "h-7 px-2", icon: "size-7" },
    },
    defaultVariants: { variant: "outline", size: "default" },
  },
);

type Props = ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof button>;

export function Button({ className, variant, size, type = "button", ...rest }: Props) {
  return <button type={type} className={cn(button({ variant, size }), className)} {...rest} />;
}
