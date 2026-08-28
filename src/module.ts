export class Parameter{
    /*
    A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    */

    public value: any;
    public name: string | undefined;

    constructor(x: any, name?: string){
        this.value = x;
        this.name = name;
        if ("requiresGrad" in x){
            this.value.requiresGrad = true;
            if (this.name != undefined){
                this.value.name = this.name;
            }
        }
    }

    update(x: any): void {
        // Update the parameter value
        this.value = x;
         if ("requiresGrad" in x){
            this.value.requiresGrad = true;
            if (this.name != undefined){
                this.value.name = this.name;
            }
        }
    }
};

export class Module{
    /*
        Modules form a tree that store parameters and other
        submodules. They make up the basis of neural network stacks.

        Attributes
        ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    */

    private _modules: Map<string, Module>;
    private _parameters: Map<string, Parameter>;
    public training: boolean; // true means training, false means evaluating

    constructor(){
        this._modules = new Map<string, Module>;
        this._parameters = new Map<string, Parameter>;
        this.training = true;

        return new Proxy(this, {
            set(target, key: string | symbol, val, rec) {
                // intercept the operation and update _modules or _parameters as needed
                if (typeof key === "string"){
                    if (val instanceof Parameter){
                        target._parameters.set(key, val);
                    }
                    else if (val instanceof Module){
                        target._modules.set(key, val);
                    }
                }

                // do the original operation
                return Reflect.set(target, key, val, rec);
            }
        });
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
            mod.eval();
        }
    }
    
    namedParameters(): [string, Parameter][]{
        /*
        Collect all the parameters of this module and its children.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.
        */

        let res : [string, Parameter][] = [];
        // Add all parameters of this module
        res.push(...this._parameters.entries());

        // Add all parameters of all of this modules children
        for (const [modName, mod] of this._modules){
            for(const [name, p] of mod.namedParameters()){
                res.push([`${modName}.${name}`, p]);
            }
        }

        return res;
    }

    parameters(): Parameter[]{
        //Enumerate over all the parameters of this module and its descendents

        // Get all parameters from this module
        let res : Parameter[] = Array.from(this._parameters.values());
        
        // Get all parameters from all of this modules children
        for (const mod of this.modules()){
            res.push(...mod.parameters());
        }

        return res;
    }

    addParameter(k: string, v: any): Parameter{
        let val = new Parameter(v, k);
        this._parameters.set(k, val);
        return val;
    }
};