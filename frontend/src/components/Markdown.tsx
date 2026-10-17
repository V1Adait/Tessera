import ReactMarkdown, { type Components } from "react-markdown";

// How each markdown element looks. Written out here (no typography plugin) so the sizes and spacing
// match the rest of the app and every line is readable.
const components: Components = {
  p: (props) => <p className="my-2 first:mt-0 last:mb-0" {...props} />,
  strong: (props) => <strong className="font-semibold" {...props} />,
  ul: (props) => <ul className="my-2 list-disc pl-5" {...props} />,
  ol: (props) => <ol className="my-2 list-decimal pl-5" {...props} />,
  li: (props) => <li className="my-0.5" {...props} />,
  h1: (props) => <h3 className="mb-1 mt-3 text-base font-semibold" {...props} />,
  h2: (props) => <h3 className="mb-1 mt-3 text-base font-semibold" {...props} />,
  h3: (props) => <h3 className="mb-1 mt-3 text-sm font-semibold" {...props} />,
  code: (props) => <code className="rounded-sm bg-border/70 px-1 py-0.5 text-[0.85em]" {...props} />,
  pre: (props) => <pre className="my-2 overflow-x-auto rounded-md bg-border/50 p-2 text-xs" {...props} />,
  blockquote: (props) => <blockquote className="my-2 border-l-2 pl-3 text-muted-foreground" {...props} />,
  a: (props) => <a className="text-primary underline" target="_blank" rel="noopener noreferrer" {...props} />,
};

/** Text from the model, shown with bold, lists and so on. Raw HTML in it is NOT rendered: it is
 * shown as text, because the model's output can carry anything from the documents it read. */
export function Markdown({ children }: { children: string }) {
  return <ReactMarkdown components={components}>{children}</ReactMarkdown>;
}
