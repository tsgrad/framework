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

class Module{
    /*
        Modules form a tree that store parameters and other
        submodules. They make up the basis of neural network stacks.

        Attributes
        ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    */

    private _modules: Map<String, Module>;
    private _parameters: Map<String, Parameter>;
    public training: Boolean; // true means training, false means evaluating

    constructor(){
        this._modules = new Map<String, Module>;
        this._parameters = new Map<String, Parameter>;
        this.training = true;
    }

    modules(): Module[]{
        return Array.from(this._modules.values());
    }

    train(): void{
        // Sets the training bool of this module and all child modules to true
        this.training = true;
        for (const mod  of this.modules()){
            mod.train();
        }
    }

    eval(): void{
        // Sets the training bool of this module and all child modules to false (meaning eval)
        this.training = false;
        for (const mod  of this.modules()){
            mod.train();
        }
    }
};