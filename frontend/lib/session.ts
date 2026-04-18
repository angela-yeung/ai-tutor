export function getThreadId(): string {
  let id = localStorage.getItem("thread_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("thread_id", id);
  }
  return id;
}

export function newSession(): string {
  const id = crypto.randomUUID();
  localStorage.setItem("thread_id", id);
  return id;
}
