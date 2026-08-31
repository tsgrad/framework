export class Queue<T>{
    private items: Record<number, T> = {};
    private head: number = 0;
    private tail: number = 0;

    push(item: T): void{
        this.items[this.tail] = item;
        this.tail++;
    }

    pop(): T | undefined {
        if (this.isEmpty())
            return undefined;

        let item = this.items[this.head];
        delete this.items[this.head];
        this.head++;
        return item;
    }

    peek(): T | undefined {
        if (this.isEmpty())
            return undefined;

        return this.items[this.head];
    }

    isEmpty(): boolean {
        return this.size() === 0;
    }

    size(): number {
        return this.tail - this.head;
    }

    clear(): void {
        this.items = {};
        this.head = 0;
        this.tail = 0;
    }
}