// Handing a file the server made to the person's computer.

/** A file name that is safe on every system, from the workspace's name: "PE diligence" ->
 * "Tessera - PE diligence - 2026-10-17.pdf". Characters Windows and macOS forbid become dashes. */
export function pdfFileName(workspaceName: string, date: Date): string {
  const printable = Array.from(workspaceName)
    .filter((ch) => ch.charCodeAt(0) >= 32) // control characters are never allowed in a file name
    .join("");
  const cleaned = printable
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[-. ]+|[-. ]+$/g, "")
    .slice(0, 60);
  const day = date.toISOString().slice(0, 10);
  return `Tessera - ${cleaned || "matrix"} - ${day}.pdf`;
}

/** Offer a blob as a download. The temporary link is removed straight away. */
export function saveBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // give the browser a moment to start the download before the address is released
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
