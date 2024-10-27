use quote::quote;
use syn::{ItemStruct, parse_macro_input};

#[proc_macro_derive(Component)]
pub fn derive_component(_item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(_item as ItemStruct);
    let name = input.ident;
    (quote! {
        impl Component for #name {}
    }).into()
}

#[proc_macro_derive(Resource)]
pub fn derive_resource(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    match input.data {
        syn::Data::Struct(_) | syn::Data::Enum(_) => {
            quote! {
                impl #impl_generics Resource for #name #ty_generics #where_clause {
                }
            }
        }
        syn::Data::Union(_) => {
            panic!("#[derive(Resource)] cannot be used on unions")
        }
    }.into()
}
#[proc_macro]
pub fn impl_system_func(_attr: proc_macro::TokenStream) -> proc_macro::TokenStream {
    const MAX_INPUT: usize = 32;

    let mut input_tys = vec![];
    let mut input_params = vec![];

    for i in 0..MAX_INPUT {
        input_tys.push(syn::Ident::new(&format!("T{i}"), proc_macro2::Span::call_site()));
        input_params.push(syn::Ident::new(&format!("t{i}"), proc_macro2::Span::call_site()));
    }

    let mut out = proc_macro2::TokenStream::new();

    out.extend(quote! {
            impl<Output, F> crate::ecs::Func<(), Output> for F
            where
                F: FnMut() -> Output + Copy + 'static,
            {
                fn execute(&mut self, _: ()) -> Output {
                    (self)()
                }
               fn boxed(&self) -> Box<dyn Func<(), Output>> {
                    Box::new(*self)
               }

            }
        
            
    });

    for i in 1..MAX_INPUT {
        let input_tys = &input_tys[..i];
        let input_params = &input_params[..i];

        out.extend(
            quote! {

            impl<#(#input_tys: Param),*, Output, F> crate::ecs::Func<(#(#input_tys,)*), Output> for F
            where
                F: FnMut(#(#input_tys),*) -> Output + Copy + 'static,
            {
                fn execute(&mut self, (#(#input_params,)*): (#(#input_tys,)*)) -> Output {
                    (self)(#(#input_params),*)
                }
               fn boxed(&self) -> Box<dyn Func<(#(#input_tys,)*), Output>> {
                    Box::new(*self)
               }

            }
    }
        );
    }

    out.into()
}



#[proc_macro]
pub fn impl_set_sequence(_attr: proc_macro::TokenStream) -> proc_macro::TokenStream {
    const MAX_INPUT: usize = 32;

    let mut input_tys = vec![];
    let mut marker_tys = vec![];
    let mut input_params = vec![];

    for i in 0..MAX_INPUT {
        input_tys.push(syn::Ident::new(&format!("T{i}"), proc_macro2::Span::call_site()));
        marker_tys.push(syn::Ident::new(&format!("M{i}"), proc_macro2::Span::call_site()));
        input_params.push(syn::Ident::new(&format!("t{i}"), proc_macro2::Span::call_site()));
    }

    let mut out = proc_macro2::TokenStream::new();

    for i in 1..MAX_INPUT {
        let input_tys = &input_tys[..i];
        let marker_tys = &marker_tys[..i];
        let input_params = &input_params[..i];

        out.extend(quote! {
            impl<#(#marker_tys,)* #(#input_tys: Sequence<#marker_tys>),*> Sequence<SetMarker<(#(#input_tys,)*), (#(#marker_tys,)*)>> for (#(#input_tys,)*) {
                type Input = (#(#input_tys::Input,)*);
                type Output =  (#(#input_tys::Output,)*);
            }
            impl<#(#marker_tys,)* #(#input_tys: Sequence<#marker_tys>),*> Capture<SetMarker<(#(#input_tys,)*), (#(#marker_tys,)*)>> for (#(#input_tys,)*) {
                fn capture(&self, pipeline: &mut Pipeline) {
                    let (#(#input_params,)*) = self;
                    #({
                        #input_params.capture(pipeline);
                    })*
                }
                fn dependencies(&self) -> Vec<Dependency> {
                    let (#(#input_params,)*) = self;
                    let mut ids = vec![];
                    #(ids.extend(#input_params.dependencies());)*
                    ids
                }

            }
        })
    }

    out.into()
}

#[proc_macro]
pub fn impl_tuple_param(_attr: proc_macro::TokenStream) -> proc_macro::TokenStream {
    const MAX_INPUT: usize = 32;

    let mut input_tys = vec![];

    for i in 0..MAX_INPUT {
        input_tys.push(syn::Ident::new(&format!("T{i}"), proc_macro2::Span::call_site()));
    }

    let mut out = proc_macro2::TokenStream::new();

    for i in 0..MAX_INPUT {
        let input_tys = &input_tys[..i];

        out.extend(quote! {
            impl<#(#input_tys: Param),*> Params for (#(#input_tys,)*) {
                fn new(state: &mut crate::ecs::State) -> Self {
                    ((#(#input_tys::new(state).expect(std::any::type_name::<#input_tys>()),)*))
                }
                fn requires() -> Prototype {
                    let mut x = vec![];
                    #(x.extend(#input_tys::requires().0);)*
                    Prototype(x.into_iter().collect())
                }
            }
        });
    }

    out.into()
}

#[proc_macro]
pub fn impl_tuple_action(_attr: proc_macro::TokenStream) -> proc_macro::TokenStream {
    const MAX_INPUT: usize = 32;

    let mut input_tys = vec![];
    let mut input_params = vec![];

    for i in 0..MAX_INPUT {
        input_tys.push(syn::Ident::new(&format!("T{i}"), proc_macro2::Span::call_site()));
        input_params.push(syn::Ident::new(&format!("t{i}"), proc_macro2::Span::call_site()));
    }

    let mut out = proc_macro2::TokenStream::new();

    for i in 0..MAX_INPUT {
        let input_tys = &input_tys[..i];
        let input_params = &input_params[..i];

        out.extend(quote! {
            impl<#(#input_tys: Action),*> Action for (#(#input_tys,)*) {
                fn act(&mut self, dispatcher: &mut Dispatcher, world: &mut World, resources: &mut Resources) {
                    let (#(#input_params,)*) = self;
                    #(#input_params.act(dispatcher, world, resources);)*
                }
            }
        });
    }

    out.into()
}
