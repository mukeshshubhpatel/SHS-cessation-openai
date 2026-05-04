import { KeyboardEvent, useEffect, useId, useMemo, useRef, useState } from "react";

type SearchableSelectProps = {
  label: string;
  value: number;
  options: number[];
  onChange: (value: number) => void;
  disabled?: boolean;
  placeholder?: string;
};

export function SearchableSelect({
  label,
  value,
  options,
  onChange,
  disabled = false,
  placeholder = "Search"
}: SearchableSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listboxId = useId();
  const labelId = useId();

  const filteredOptions = useMemo(() => {
    const term = search.trim();
    if (!term) return options;
    return options.filter((option) => String(option).includes(term));
  }, [options, search]);

  useEffect(() => {
    if (!open) return;

    setActiveIndex(0);
    const frame = window.requestAnimationFrame(() => {
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    });

    return () => window.cancelAnimationFrame(frame);
  }, [open]);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent | TouchEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
        setSearch("");
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("touchstart", handlePointerDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("touchstart", handlePointerDown);
    };
  }, []);

  useEffect(() => {
    if (activeIndex >= filteredOptions.length) {
      setActiveIndex(0);
    }
  }, [activeIndex, filteredOptions.length]);

  function closeMenu() {
    setOpen(false);
    setSearch("");
  }

  function commitSelection(nextValue: number) {
    onChange(nextValue);
    closeMenu();
  }

  function onTriggerKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (disabled) return;

    if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setOpen(true);
    }
  }

  function onSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeMenu();
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => Math.min(current + 1, Math.max(filteredOptions.length - 1, 0)));
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((current) => Math.max(current - 1, 0));
      return;
    }

    if (event.key === "Enter" && filteredOptions[activeIndex] !== undefined) {
      event.preventDefault();
      commitSelection(filteredOptions[activeIndex]);
    }
  }

  return (
    <div className={`searchable-select ${open ? "open" : ""} ${disabled ? "disabled" : ""}`} ref={rootRef}>
      <div className="field-label" id={labelId}>
        {label}
      </div>
      <button
        type="button"
        className="select-trigger"
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-controls={listboxId}
        aria-labelledby={labelId}
        disabled={disabled}
        onClick={() => {
          setOpen((current) => !current);
          if (open) {
            setSearch("");
          }
        }}
        onKeyDown={onTriggerKeyDown}
      >
        <span>{value}</span>
        <svg viewBox="0 0 20 20" className="select-caret" aria-hidden="true">
          <path d="M5.25 7.5L10 12.25L14.75 7.5" />
        </svg>
      </button>

      <div
        className="select-dropdown"
        role="dialog"
        aria-labelledby={labelId}
        aria-hidden={!open}
      >
        <div className="select-search-wrap">
          <input
            ref={searchInputRef}
            type="text"
            inputMode="numeric"
            className="select-search"
            value={search}
            onChange={(event) => setSearch(event.target.value.replace(/[^\d]/g, ""))}
            onKeyDown={onSearchKeyDown}
            placeholder={`${placeholder} age`}
          />
        </div>

        <div className="select-options" role="listbox" id={listboxId} aria-labelledby={labelId}>
          {filteredOptions.length ? (
            filteredOptions.map((option, index) => (
              <button
                type="button"
                role="option"
                aria-selected={option === value}
                key={option}
                className={`select-option ${option === value ? "selected" : ""} ${index === activeIndex ? "active" : ""}`}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => commitSelection(option)}
              >
                <span className="select-option-label">{option}</span>
                {option === value ? (
                  <span className="select-check" aria-hidden="true">
                    <svg viewBox="0 0 20 20" className="select-check-icon">
                      <path d="M5 10.5L8.2 13.6L15 6.8" />
                    </svg>
                  </span>
                ) : null}
              </button>
            ))
          ) : (
            <div className="select-empty">No matching ages</div>
          )}
        </div>
      </div>
    </div>
  );
}
