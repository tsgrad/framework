class Parameter{
    /*
    A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    */

    public value: any;
    public name: String | undefined;

    constructor(x: any, name?: String){
        this.value = x;
        this.name = name;
        if ("requires_grad" in x){
            this.value.requires_grad = true;
            if (this.name != undefined){
                this.value.name = this.name;
            }
        }
    }

    update(x: any): void {
        // Update the parameter value
        this.value = x;
         if ("requires_grad" in x){
            this.value.requires_grad = true;
            if (this.name != undefined){
                this.value.name = this.name;
            }
        }
    }
};